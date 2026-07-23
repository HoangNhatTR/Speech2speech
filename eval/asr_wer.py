"""Đo WER (Word Error Rate) của STT_BACKEND đang cấu hình trong .env, trên tập câu
eval/testset.py — theo domain (chao_hoi/ngan_hang/so_thoi_gian/code_switch/...) và theo
mức nhiễu mô phỏng (xem eval/testset.py::NOISE_LEVELS_DB), để trả lời đúng câu hỏi "hệ
thống có hoạt động tốt với MỌI trường hợp data không" thay vì chỉ một con số WER gộp có
thể che khuất domain yếu. Xem docstring eval/testset.py về phương pháp (TTS round-trip
qua gTTS + cộng nhiễu trắng mô phỏng, KHÔNG phải giọng người/nhiễu môi trường thật — đọc
kỹ trước khi diễn giải kết quả).

Chạy: python -m eval.asr_wer
Kết quả: in bảng ra console + ghi eval/results/asr_wer_<timestamp>.json
"""

import asyncio
import io
import json
import time
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from gtts import gTTS

from bot import build_stt
from eval.testset import TEST_ITEMS, NOISE_LEVELS_DB

load_dotenv(override=True)

WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _synthesize_samples(text: str) -> tuple[np.ndarray, int]:
    """gTTS -> mp3 (trong bộ nhớ) -> mono float32 PCM samples."""
    mp3_buf = io.BytesIO()
    gTTS(text=text, lang="vi").write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    samples, sample_rate = sf.read(mp3_buf, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate


def add_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Cộng nhiễu trắng Gaussian để đạt SNR mục tiêu (xấp xỉ, dựa trên công suất trung
    bình toàn đoạn) — mô phỏng thô độ suy giảm, không thay được nhiễu môi trường thật.
    seed cố định theo (câu, mức nhiễu) để kết quả tái lập được giữa các lần chạy (so
    sánh model A/B phải cùng một bản nhiễu, không phải nhiễu ngẫu nhiên mỗi lần)."""
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(samples**2)) or 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, noise_power**0.5, size=samples.shape).astype(np.float32)
    return np.clip(samples + noise, -1.0, 1.0)


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = (samples * 32767.0).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return wav_buf.getvalue()


async def transcribe(stt, wav_bytes: bytes) -> str:
    text_parts = []
    async for frame in stt.run_stt(wav_bytes):
        text = getattr(frame, "text", None)
        if text:
            text_parts.append(text)
    return " ".join(text_parts).strip()


def _wer(reference: str, hypothesis: str) -> float:
    return jiwer.process_words(
        reference, hypothesis or "", reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM
    ).wer


def _corpus_wer(rows: list[dict]) -> float:
    return jiwer.wer(
        [r["reference"] for r in rows],
        [r["hypothesis"] or "" for r in rows],
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    )


async def run() -> dict:
    """Chạy toàn bộ tập test qua mọi mức nhiễu, trả về summary dict (dùng lại được từ
    eval/run_benchmarks.py mà không cần parse file JSON)."""
    stt = build_stt()
    backend_name = type(stt).__name__

    rows = []
    for item in TEST_ITEMS:
        clean_samples, sample_rate = _synthesize_samples(item.text)
        for noise_db in NOISE_LEVELS_DB:
            if noise_db is None:
                samples, noise_label = clean_samples, "clean"
            else:
                samples = add_noise(clean_samples, noise_db, seed=hash((item.text, noise_db)) % (2**32))
                noise_label = f"{noise_db:g}db"

            wav_bytes = _to_wav_bytes(samples, sample_rate)
            t0 = time.time()
            hypothesis = await transcribe(stt, wav_bytes)
            elapsed_ms = (time.time() - t0) * 1000

            rows.append(
                {
                    "reference": item.text,
                    "hypothesis": hypothesis,
                    "domain": item.domain,
                    "code_switch": item.code_switch,
                    "noise": noise_label,
                    "wer": _wer(item.text, hypothesis),
                    "elapsed_ms": round(elapsed_ms, 1),
                }
            )

    overall_wer = _corpus_wer(rows)
    avg_latency_ms = sum(r["elapsed_ms"] for r in rows) / len(rows)

    by_domain: dict[str, float] = {}
    by_noise: dict[str, float] = {}
    groups_domain: dict[str, list[dict]] = defaultdict(list)
    groups_noise: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups_domain[row["domain"]].append(row)
        groups_noise[row["noise"]].append(row)
    for domain, group_rows in groups_domain.items():
        by_domain[domain] = _corpus_wer(group_rows)
    for noise_label, group_rows in groups_noise.items():
        by_noise[noise_label] = _corpus_wer(group_rows)

    code_switch_rows = [r for r in rows if r["code_switch"]]
    code_switch_wer = _corpus_wer(code_switch_rows) if code_switch_rows else None

    return {
        "backend": backend_name,
        "overall_wer": overall_wer,
        "by_domain_wer": by_domain,
        "by_noise_wer": by_noise,
        "code_switch_wer": code_switch_wer,
        "avg_latency_ms": avg_latency_ms,
        "n": len(rows),
        "rows": rows,
    }


async def main() -> None:
    summary = await run()
    print(f"STT backend: {summary['backend']}\n")

    for row in summary["rows"]:
        print(
            f"[{row['wer']:.2f} WER, {row['elapsed_ms']:.0f}ms] domain={row['domain']:<12} "
            f"noise={row['noise']:<6} {row['reference']!r} -> {row['hypothesis']!r}"
        )

    print(f"\nWER theo domain:")
    for domain, wer in sorted(summary["by_domain_wer"].items()):
        print(f"  {domain:<14} {wer:.3f}")

    print(f"\nWER theo mức nhiễu:")
    for noise_label, wer in sorted(summary["by_noise_wer"].items()):
        print(f"  {noise_label:<8} {wer:.3f}")

    if summary["code_switch_wer"] is not None:
        print(f"\nWER code-switch VN-EN: {summary['code_switch_wer']:.3f}")

    print(
        f"\nCorpus WER tổng (mọi domain+mức nhiễu gộp): {summary['overall_wer']:.3f}  |  "
        f"Avg decode latency: {summary['avg_latency_ms']:.0f}ms  |  N={summary['n']}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"asr_wer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
