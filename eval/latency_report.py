"""Tổng hợp p50/p95 từ eval/results/latency_log.jsonl (được bot.py ghi mỗi khi có hội
thoại voice thật). Nhóm theo (metric, stt_backend, llm_backend, tts_backend) để so sánh
latency giữa các cấu hình backend khác nhau.

Chạy: python -m eval.latency_report
"""

import json
from collections import defaultdict
from pathlib import Path

from eval.latency_log import LOG_PATH


def percentile(values: list, p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def main() -> None:
    if not LOG_PATH.exists():
        print(f"Chưa có log: {LOG_PATH}")
        print("Log được ghi khi chạy hội thoại voice thật qua gateway/main.py (bot.py).")
        return

    groups = defaultdict(list)
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (
                record["metric"],
                record["stt_backend"],
                record["llm_backend"],
                record["tts_backend"],
            )
            groups[key].append(record["value_ms"])

    print(f"{'metric':<18} {'stt':<8} {'llm':<8} {'tts':<8} {'n':>4} {'p50':>8} {'p95':>8} {'mean':>8}")
    for (metric, stt, llm, tts), values in sorted(groups.items()):
        p50 = percentile(values, 0.5)
        p95 = percentile(values, 0.95)
        mean = sum(values) / len(values)
        print(
            f"{metric:<18} {stt:<8} {llm:<8} {tts:<8} {len(values):>4} "
            f"{p50:>7.0f}ms {p95:>7.0f}ms {mean:>7.0f}ms"
        )


if __name__ == "__main__":
    main()
