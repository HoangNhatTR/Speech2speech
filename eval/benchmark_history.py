"""Lưu trữ lịch sử các lần chạy benchmark (JSONL, append-only) — dashboard đọc file này
để vẽ xu hướng theo thời gian (quy ước "liên tục phải có benchmark với SOTA"). Tách
riêng khỏi từng script eval/*.py vì lịch sử này gộp nhiều loại benchmark (WER, tool-call,
duplex-bench, latency) vào cùng một dòng thời gian, còn mỗi script vẫn tự ghi file JSON
chi tiết riêng của nó như trước (không phá vỡ hành vi cũ)."""

import json
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent / "results" / "benchmark_history.jsonl"


def append(entry: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_all() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    entries = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def latest() -> dict | None:
    entries = read_all()
    return entries[-1] if entries else None
