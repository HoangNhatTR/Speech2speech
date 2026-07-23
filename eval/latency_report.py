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


def summarize() -> list[dict]:
    """Đọc eval/results/latency_log.jsonl và trả về danh sách nhóm (metric, backend
    config) kèm p50/p95/mean/n — dùng lại được từ eval/run_benchmarks.py và
    gateway/dashboard_api.py mà không phải parse JSONL hai lần."""
    if not LOG_PATH.exists():
        return []

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

    rows = []
    for (metric, stt, llm, tts), values in sorted(groups.items()):
        rows.append(
            {
                "metric": metric,
                "stt_backend": stt,
                "llm_backend": llm,
                "tts_backend": tts,
                "n": len(values),
                "p50_ms": percentile(values, 0.5),
                "p95_ms": percentile(values, 0.95),
                "mean_ms": sum(values) / len(values),
            }
        )
    return rows


def main() -> None:
    rows = summarize()
    if not rows:
        print(f"Chưa có log: {LOG_PATH}")
        print("Log được ghi khi chạy hội thoại voice thật qua gateway/main.py (bot.py).")
        return

    print(f"{'metric':<18} {'stt':<8} {'llm':<8} {'tts':<8} {'n':>4} {'p50':>8} {'p95':>8} {'mean':>8}")
    for row in rows:
        print(
            f"{row['metric']:<18} {row['stt_backend']:<8} {row['llm_backend']:<8} {row['tts_backend']:<8} "
            f"{row['n']:>4} {row['p50_ms']:>7.0f}ms {row['p95_ms']:>7.0f}ms {row['mean_ms']:>7.0f}ms"
        )


if __name__ == "__main__":
    main()
