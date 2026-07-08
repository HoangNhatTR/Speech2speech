"""Ghi log latency có cấu trúc (JSONL) — dùng chung bởi bot.py (khi có hội thoại thật)
và eval/latency_report.py (để tổng hợp p50/p95). Trước đây latency chỉ in ra console
qua logger.info — vẫn còn (dễ đọc khi debug trực tiếp), giờ thêm ghi file để tổng hợp
được theo thời gian và theo backend đang dùng.
"""

import json
import os
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "results" / "latency_log.jsonl"


def append(metric: str, value_ms: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "metric": metric,
        "value_ms": value_ms,
        "stt_backend": os.getenv("STT_BACKEND", "cloud"),
        "llm_backend": os.getenv("LLM_BACKEND", "cloud"),
        "tts_backend": os.getenv("TTS_BACKEND", "cloud"),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
