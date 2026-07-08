"""Đo WER (Word Error Rate) của STT_BACKEND đang cấu hình trong .env, trên tập câu
eval/testset.py. Xem docstring trong testset.py về phương pháp (TTS round-trip qua
gTTS, không phải giọng người thật — đọc kỹ trước khi diễn giải kết quả).

Chạy: python -m eval.asr_wer
Kết quả: in bảng ra console + ghi eval/results/asr_wer_<timestamp>.json
"""

import asyncio
import io
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from gtts import gTTS

from bot import build_stt
from eval.testset import SENTENCES

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


def synthesize_reference_wav(text: str) -> bytes:
    """gTTS -> mp3 (trong bộ nhớ) -> WAV PCM16 mono, dùng làm audio đầu vào cho STT."""
    mp3_buf = io.BytesIO()
    gTTS(text=text, lang="vi").write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    samples, sample_rate = sf.read(mp3_buf, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
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


async def main() -> None:
    stt = build_stt()
    backend_name = type(stt).__name__
    print(f"STT backend: {backend_name}\n")

    rows = []
    for text in SENTENCES:
        wav_bytes = synthesize_reference_wav(text)
        t0 = time.time()
        hypothesis = await transcribe(stt, wav_bytes)
        elapsed_ms = (time.time() - t0) * 1000

        result = jiwer.process_words(
            text, hypothesis or "", reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM
        )
        rows.append(
            {
                "reference": text,
                "hypothesis": hypothesis,
                "wer": result.wer,
                "elapsed_ms": round(elapsed_ms, 1),
            }
        )
        print(f"[{result.wer:.2f} WER, {elapsed_ms:.0f}ms] {text!r} -> {hypothesis!r}")

    corpus_wer = jiwer.wer(
        [r["reference"] for r in rows],
        [r["hypothesis"] or "" for r in rows],
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    )
    avg_latency = sum(r["elapsed_ms"] for r in rows) / len(rows)
    print(f"\nCorpus WER: {corpus_wer:.3f}  |  Avg decode latency: {avg_latency:.0f}ms  |  N={len(rows)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"asr_wer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps(
            {
                "backend": backend_name,
                "corpus_wer": corpus_wer,
                "avg_latency_ms": avg_latency,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Đã ghi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
