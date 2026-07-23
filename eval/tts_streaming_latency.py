"""Đo TTFB và RTF của VieNeu-TTS streaming qua đúng HTTP endpoint production.

Chạy TTS server trước, rồi:
    python -m eval.tts_streaming_latency
    python -m eval.tts_streaming_latency --assert-realtime

`--assert-realtime` trả exit code 1 nếu p50 time-to-first-audio vượt 500ms hoặc p50 RTF
vượt 1.0. Không ghi file/kết quả benchmark vào repo.
"""

import argparse
import asyncio
import statistics
import time

import aiohttp

DEFAULT_TEXTS = [
    "Xin chào bạn.",
    "Hôm nay thời tiết thật dễ chịu.",
    "Tôi có thể giúp gì cho bạn?",
]


async def measure(session: aiohttp.ClientSession, url: str, text: str) -> dict:
    started_at = time.perf_counter()
    first_at = None
    total_bytes = 0

    async with session.post(f"{url.rstrip('/')}/synthesize/stream", json={"text": text}) as response:
        response.raise_for_status()
        sample_rate = int(response.headers["X-Audio-Sample-Rate"])
        async for chunk in response.content.iter_chunked(16_384):
            if first_at is None:
                first_at = time.perf_counter()
            total_bytes += len(chunk)

    ended_at = time.perf_counter()
    if first_at is None or total_bytes == 0:
        raise RuntimeError("TTS server trả response rỗng")
    audio_s = total_bytes / (2 * sample_rate)  # PCM16 mono
    return {
        "text": text,
        "ttfb_ms": (first_at - started_at) * 1000,
        "total_ms": (ended_at - started_at) * 1000,
        "audio_s": audio_s,
        "rtf": (ended_at - started_at) / audio_s,
    }


async def run(url: str, texts: list[str]) -> list[dict]:
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        rows = []
        for text in texts:
            row = await measure(session, url, text)
            rows.append(row)
            print(
                f"TTFB={row['ttfb_ms']:>5.0f}ms  RTF={row['rtf']:.2f}  "
                f"audio={row['audio_s']:.2f}s  {text}"
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--assert-realtime", action="store_true")
    args = parser.parse_args()

    rows = asyncio.run(run(args.url, DEFAULT_TEXTS))
    p50_ttfb = statistics.median(row["ttfb_ms"] for row in rows)
    p50_rtf = statistics.median(row["rtf"] for row in rows)
    print(f"\np50 TTFB={p50_ttfb:.0f}ms, p50 RTF={p50_rtf:.2f}")

    if args.assert_realtime and (p50_ttfb > 500 or p50_rtf > 1.0):
        raise SystemExit(
            "Chưa đạt real-time: cần p50 TTFB <= 500ms và p50 RTF <= 1.0. "
            "Thử VIENEU_DEVICE=cuda và restart TTS server."
        )


if __name__ == "__main__":
    main()
