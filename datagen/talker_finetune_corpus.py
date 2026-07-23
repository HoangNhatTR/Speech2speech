"""Chuẩn bị dữ liệu audio-text cho fine-tune Talker (Qwen3-Omni) nói tiếng Việt —
Giai đoạn 3 hướng (a), xem docs/roadmap.md mục 2 và mục 8.

Đây LÀ bước duy nhất của Giai đoạn 3 làm được ngay trên máy hiện tại: máy GB10 dùng
chung không đủ quy mô cho bước train thật (roadmap mục 10 ước tính cần ~8×A100 80GB
vài tuần, kể cả chỉ LoRA trên Talker) — xem docs/platform-architecture.md để biết chi
tiết đo đạc thật. Script này KHÔNG train gì cả, chỉ tải + chuẩn hoá dữ liệu thành
manifest JSONL sẵn sàng cho bước train khi có GPU đủ (thuê cloud hoặc khác).

Ba bộ dữ liệu hỗ trợ (đăng ký trong DATASETS bên dưới):

  vivos       — AILAB-VNUHCM/vivos, CÔNG KHAI (không cần xin quyền), ~15h đọc, dùng để
                validate pipeline này chạy đúng ngay hôm nay.
  vivoice     — capleaf/viVoice, 1000h+, 24kHz, BỊ GATE trên HuggingFace (yêu cầu email
                trường/công ty, không nhận email cá nhân phổ biến). Phải tự xin quyền
                tại https://huggingface.co/datasets/capleaf/viVoice rồi
                `hf auth login` bằng token của tài khoản đã được duyệt TRƯỚC khi chạy
                --dataset vivoice. Lưu ý roadmap: text chưa chuẩn hoá (số, viết tắt).
  phoaudiobook — thivux/phoaudiobook, 941h/735 speaker, có speaker ID, cũng BỊ GATE
                (https://huggingface.co/datasets/thivux/phoaudiobook), cùng yêu cầu xin
                quyền + `hf auth login` như trên.

Vì viVoice/PhoAudiobook rất lớn (168GB+ nén), script load qua streaming=True — không
tải nguyên bộ, chỉ kéo đủ số phút/giờ audio cần bằng --max-duration-hours.

Chuẩn hoá áp dụng cho mọi nguồn:
  - Resample audio về 16kHz mono (khớp sampling_rate mà Qwen3-Omni Thinker audio
    encoder yêu cầu — xem processing_qwen3_omni_moe.py) — dùng cho input phía
    Thinker khi precompute hidden-states sau này. Audio gốc (chưa resample) KHÔNG
    được giữ lại vì mục tiêu bước này là chuẩn bị input cho Thinker, không phải
    train Code2Wav (decoder tái tạo audio) — bước đó cần giữ audio ở sample rate
    gốc của model, sẽ bổ sung script riêng khi thiết kế xong training loop thật
    (hiện chưa có recipe chính thức, xem docs/roadmap.md mục 2 phần (a)).
  - Text chỉ strip + gộp khoảng trắng thừa — CHƯA chuẩn hoá số/viết tắt/dấu câu đầy
    đủ (việc này roadmap tự ghi nhận là vấn đề của viVoice, cần thêm bước riêng nếu
    dùng nguồn này).

Chạy thử ngay (không cần token, ~vài phút):
    .venv-tts/bin/python -m datagen.talker_finetune_corpus --dataset vivos \\
        --max-duration-hours 0.2 --out data/talker_corpus

Chạy với dữ liệu đã xin quyền:
    .venv-tts/bin/python -m datagen.talker_finetune_corpus --dataset vivoice \\
        --max-duration-hours 5 --out data/talker_corpus
"""

import argparse
import re
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
from loguru import logger

TARGET_SR = 16000


@dataclass
class Example:
    text: str
    samples: np.ndarray
    sample_rate: int
    speaker: Optional[str]


@dataclass
class DatasetSpec:
    hf_id: str
    split: str
    text_field: str
    audio_field: str
    speaker_field: Optional[str]
    gated: bool


# vivos dùng cấu trúc file thô (tar.gz + prompts.txt), KHÔNG qua `datasets` — bản mới
# của thư viện `datasets` (5.x) đã bỏ hỗ trợ dataset loading script mà vivos.py trong
# repo HF dùng (lỗi thật gặp: "Dataset scripts are no longer supported"). vivoice và
# phoaudiobook ở dạng parquet hiện đại nên load bình thường qua `datasets` streaming.
DATASETS = {
    "vivos": DatasetSpec(
        hf_id="AILAB-VNUHCM/vivos",
        split="train",
        text_field="",
        audio_field="",
        speaker_field=None,
        gated=False,
    ),
    "vivoice": DatasetSpec(
        hf_id="capleaf/viVoice",
        split="train",
        text_field="text",
        audio_field="audio",
        speaker_field="channel",
        gated=True,
    ),
    "phoaudiobook": DatasetSpec(
        hf_id="thivux/phoaudiobook",
        split="train",
        text_field="text",
        audio_field="audio",
        speaker_field="speaker_id",
        gated=True,
    ),
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    import librosa

    return librosa.resample(
        audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    )


def _iter_vivos(split: str) -> Iterator[Example]:
    """Tải vivos.tar.gz (~1.4GB, cache qua huggingface_hub), giải nén ra đĩa MỘT LẦN
    rồi đọc file thường — cấu trúc: vivos/{split}/prompts.txt (UTT_ID TRANSCRIPT) +
    vivos/{split}/waves/<SPEAKER>/<UTT_ID>.wav.

    Bản đầu tiên của hàm này gọi `tar.extractfile(name)` theo từng file riêng lẻ mà
    KHÔNG giải nén trước — đo thật thấy quá chậm (~2.4s/câu, ước tính ~8 tiếng cho
    11.660 câu của vivos) vì gzip không hỗ trợ seek ngẫu nhiên, mỗi lần tra cứu một
    file phải quét lại từ đầu stream nén. Giải nén hết ra đĩa một lần (một lượt giải
    nén tuần tự, nhanh) rồi đọc bằng filesystem thường nhanh hơn nhiều.
    """
    from huggingface_hub import hf_hub_download

    tar_path = hf_hub_download(
        repo_id="AILAB-VNUHCM/vivos",
        repo_type="dataset",
        filename="data/vivos.tar.gz",
    )

    extract_dir = Path(tar_path).parent / "extracted"
    split_dir = extract_dir / "vivos" / split
    if not (split_dir / "prompts.txt").exists():
        logger.info("Giải nén vivos.tar.gz ra đĩa (một lần, khoảng 1-2 phút)...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

    prompts = {}
    with open(split_dir / "prompts.txt", encoding="utf-8") as f:
        for line in f:
            utt_id, _, transcript = line.partition(" ")
            if utt_id:
                prompts[utt_id] = transcript.strip()

    for utt_id, transcript in prompts.items():
        speaker = utt_id.split("_")[0]
        wav_path = split_dir / "waves" / speaker / f"{utt_id}.wav"
        if not wav_path.exists():
            continue
        samples, sr = sf.read(wav_path, dtype="float32")
        yield Example(text=transcript, samples=samples, sample_rate=sr, speaker=speaker)


def _iter_hf_streaming(spec: DatasetSpec) -> Iterator[Example]:
    if spec.gated:
        logger.warning(
            f"{spec.hf_id} bị gate trên HuggingFace — nếu lệnh dưới đây báo lỗi "
            f"401/403, cần xin quyền tại https://huggingface.co/datasets/{spec.hf_id} "
            f"rồi `hf auth login` bằng token của tài khoản đã được duyệt."
        )

    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, split=spec.split, streaming=True)
    for example in ds:
        audio = example[spec.audio_field]
        yield Example(
            text=example[spec.text_field],
            samples=np.asarray(audio["array"]),
            sample_rate=audio["sampling_rate"],
            speaker=example.get(spec.speaker_field) if spec.speaker_field else None,
        )


def build_corpus(dataset_name: str, max_duration_hours: float, out_dir: Path) -> Path:
    spec = DATASETS[dataset_name]
    examples = _iter_vivos(spec.split) if dataset_name == "vivos" else _iter_hf_streaming(spec)

    wav_dir = out_dir / dataset_name / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / dataset_name / "manifest.jsonl"

    max_duration_sec = max_duration_hours * 3600
    collected_sec = 0.0
    n_examples = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for example in examples:
            if collected_sec >= max_duration_sec:
                break

            text = normalize_text(example.text)
            if not text:
                continue

            samples_16k = resample(example.samples, example.sample_rate, TARGET_SR)
            duration_sec = len(samples_16k) / TARGET_SR

            wav_path = wav_dir / f"{n_examples:07d}.wav"
            sf.write(wav_path, samples_16k, TARGET_SR, subtype="PCM_16")

            record = {
                "id": n_examples,
                "source": dataset_name,
                "audio_path": str(wav_path),
                "text": text,
                "duration_sec": round(duration_sec, 3),
                "sample_rate": TARGET_SR,
                "speaker": example.speaker,
            }
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            collected_sec += duration_sec
            n_examples += 1
            if n_examples % 50 == 0:
                logger.info(
                    f"[{dataset_name}] {n_examples} câu, "
                    f"{collected_sec / 3600:.2f}h / {max_duration_hours}h"
                )

    logger.info(
        f"[{dataset_name}] Xong: {n_examples} câu, {collected_sec / 3600:.3f}h "
        f"-> {manifest_path}"
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--max-duration-hours", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/talker_corpus"))
    args = parser.parse_args()

    try:
        build_corpus(args.dataset, args.max_duration_hours, args.out)
    except Exception as e:
        logger.error(f"Lỗi khi build corpus '{args.dataset}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
