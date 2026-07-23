"""Compare Vietnamese ASR before and after Mimi reconstruction.

Run this in the main ``.venv`` after ``eval.mimi_codec`` produced a 24 kHz WAV. It
uses the same local Zipformer as the Anchor path and reports absolute WER degradation,
which is the codec go/no-go metric in ``docs/ke-hoach-cong-viec-train-test-s2s.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import jiwer
from dotenv import load_dotenv

from selfhost.asr import ZipformerVietnameseSTTService

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


async def _transcribe(stt: ZipformerVietnameseSTTService, path: Path) -> tuple[str, float]:
    started = time.perf_counter()
    parts: list[str] = []
    errors: list[str] = []
    async for frame in stt.run_stt(path.read_bytes()):
        text = getattr(frame, "text", None)
        error = getattr(frame, "error", None)
        if text:
            parts.append(text)
        if error:
            errors.append(error)
    if errors:
        raise RuntimeError("; ".join(errors))
    return " ".join(parts).strip(), (time.perf_counter() - started) * 1000


def _wer(reference: str, hypothesis: str) -> float:
    return jiwer.process_words(
        reference,
        hypothesis or "",
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    ).wer


async def run(original: Path, reconstructed: Path, transcript: str) -> dict:
    model_dir = os.getenv(
        "ASR_MODEL_DIR", "models/asr-vi/sherpa-onnx-zipformer-vi-int8-2025-04-20"
    )
    stt = ZipformerVietnameseSTTService(model_dir=model_dir)
    original_text, original_latency_ms = await _transcribe(stt, original)
    reconstructed_text, reconstructed_latency_ms = await _transcribe(stt, reconstructed)
    original_wer = _wer(transcript, original_text)
    reconstructed_wer = _wer(transcript, reconstructed_text)
    return {
        "original": str(original),
        "reconstructed": str(reconstructed),
        "reference": transcript,
        "original_hypothesis": original_text,
        "reconstructed_hypothesis": reconstructed_text,
        "original_wer": round(original_wer, 6),
        "reconstructed_wer": round(reconstructed_wer, 6),
        "wer_delta_absolute": round(reconstructed_wer - original_wer, 6),
        "codec_gate_max_delta": 0.02,
        "codec_gate_pass": reconstructed_wer - original_wer <= 0.02,
        "original_asr_latency_ms": round(original_latency_ms, 3),
        "reconstructed_asr_latency_ms": round(reconstructed_latency_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    result = asyncio.run(run(args.original, args.reconstructed, args.transcript))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["codec_gate_pass"] else 2)


if __name__ == "__main__":
    main()
