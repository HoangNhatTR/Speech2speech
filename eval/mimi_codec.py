"""Vietnamese Mimi codec reconstruction benchmark (no Moshi LM training).

This command intentionally lives in the optional ``.venv-s2s``. It downloads only the
official Mimi checkpoint on first use, resamples input to 24 kHz, performs streaming
encode and streaming decode in exact ``mimi.frame_size`` chunks, then writes a WAV and
JSON metrics. It never mutates the realtime Gateway configuration.

Example:
    .venv-s2s/bin/python -m eval.mimi_codec \
        --input path/to/vietnamese.wav \
        --output eval/results/mimi_reconstructed.wav \
        --result-json eval/results/mimi_codec.json \
        --device cuda

Listen to the result and run the normal ASR over original/reconstructed audio before
deciding whether Mimi needs Vietnamese codec adaptation. Waveform SNR is deliberately
not reported: a neural codec may introduce delay/phase changes that make unaligned SNR
misleading even when speech quality is good.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 24_000


def _load_optional_runtime():
    try:
        import torch
        import torchaudio.functional as audio_functional
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu Mimi runtime. Tạo .venv-s2s và cài requirements-s2s.txt; "
            "Torch/Torchaudio phải khớp CUDA của máy."
        ) from exc
    return torch, audio_functional, hf_hub_download, loaders


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError(f"Audio rỗng: {path}")
    return samples.mean(axis=1), sample_rate


def run_benchmark(
    input_path: Path,
    output_path: Path,
    *,
    device: str,
    hf_repo: str | None,
    num_codebooks: int,
    warmup_frames: int,
) -> dict:
    torch, audio_functional, hf_hub_download, loaders = _load_optional_runtime()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda nhưng torch.cuda.is_available() = false")

    samples, source_sample_rate = _read_mono(input_path)
    wav = torch.from_numpy(samples).view(1, 1, -1)
    if source_sample_rate != TARGET_SAMPLE_RATE:
        wav = audio_functional.resample(wav, source_sample_rate, TARGET_SAMPLE_RATE)
    original_samples = wav.shape[-1]

    repo = hf_repo or loaders.DEFAULT_REPO
    started_load = time.perf_counter()
    weight = hf_hub_download(repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(weight, device=device)
    mimi.set_num_codebooks(num_codebooks)
    load_seconds = time.perf_counter() - started_load

    frame_size = int(mimi.frame_size)
    warmup_seconds = 0.0
    if warmup_frames:
        warmup = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        warmup_started = time.perf_counter()
        with torch.no_grad(), mimi.streaming(batch_size=1):
            for _ in range(warmup_frames):
                warmup_codes = mimi.encode(warmup)
                _ = mimi.decode(warmup_codes)
        if device == "cuda":
            torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started

    padded_samples = math.ceil(original_samples / frame_size) * frame_size
    if padded_samples > original_samples:
        wav = torch.nn.functional.pad(wav, (0, padded_samples - original_samples))
    wav = wav.to(device)

    codes = []
    encode_started = time.perf_counter()
    with torch.no_grad(), mimi.streaming(batch_size=1):
        for offset in range(0, padded_samples, frame_size):
            frame = wav[:, :, offset : offset + frame_size]
            code = mimi.encode(frame)
            if code.shape[-1] != 1:
                raise RuntimeError(f"Mimi trả code shape không streaming: {tuple(code.shape)}")
            codes.append(code.cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started

    decoded_chunks = []
    first_decoded_seconds = None
    decode_started = time.perf_counter()
    with torch.no_grad(), mimi.streaming(batch_size=1):
        for code in codes:
            decoded = mimi.decode(code.to(device))
            if first_decoded_seconds is None:
                if device == "cuda":
                    torch.cuda.synchronize()
                first_decoded_seconds = time.perf_counter() - decode_started
            decoded_chunks.append(decoded.cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started

    reconstructed = torch.cat(decoded_chunks, dim=-1)[0, 0, :original_samples]
    reconstructed_np = reconstructed.float().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, reconstructed_np, TARGET_SAMPLE_RATE, subtype="PCM_16")

    duration_seconds = original_samples / TARGET_SAMPLE_RATE
    total_codec_seconds = encode_seconds + decode_seconds
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "source_sample_rate": source_sample_rate,
        "codec_sample_rate": TARGET_SAMPLE_RATE,
        "duration_seconds": round(duration_seconds, 6),
        "frame_size_samples": frame_size,
        "frame_duration_ms": round(frame_size * 1000 / TARGET_SAMPLE_RATE, 3),
        "frames": len(codes),
        "num_codebooks": num_codebooks,
        "device": device,
        "hf_repo": repo,
        "model_load_seconds": round(load_seconds, 6),
        "warmup_frames": warmup_frames,
        "warmup_seconds": round(warmup_seconds, 6),
        "encode_seconds": round(encode_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
        "first_decoded_frame_seconds": round(first_decoded_seconds or 0.0, 6),
        "codec_rtf": round(total_codec_seconds / duration_seconds, 6),
        "padded_samples": padded_samples - original_samples,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hf-repo", help="Mặc định dùng moshi.models.loaders.DEFAULT_REPO")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=4,
        help="Compile/warm Mimi trước khi đo; 4 frame giống server chính thức",
    )
    args = parser.parse_args()

    if not 1 <= args.num_codebooks <= 32:
        parser.error("--num-codebooks phải trong khoảng 1..32; Moshi dùng 8")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames không được âm")
    result = run_benchmark(
        args.input,
        args.output,
        device=args.device,
        hf_repo=args.hf_repo,
        num_codebooks=args.num_codebooks,
        warmup_frames=args.warmup_frames,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
