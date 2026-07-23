"""Orchestrator benchmark — chạy theo quy ước "liên tục phải có benchmark với SOTA":
gộp duplex-bench (luôn chạy, miễn phí), latency report (đọc log có sẵn, miễn phí), và
tuỳ chọn ASR WER / tool-call accuracy (tốn API/GPU thật, mặc định TẮT — bật bằng cờ) vào
một lần chạy, so với eval/sota_reference.json, rồi ghi một dòng vào
eval/results/benchmark_history.jsonl để dashboard vẽ xu hướng.

Chạy:
  python -m eval.run_benchmarks                      # chỉ phần miễn phí (duplex-bench + latency)
  python -m eval.run_benchmarks --with-asr            # + đo WER (gọi STT_BACKEND thật)
  python -m eval.run_benchmarks --with-tool-call      # + đo tool-call accuracy (gọi LLM_BACKEND thật)
  python -m eval.run_benchmarks --with-asr --with-tool-call
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from eval import benchmark_history, latency_report
from eval.duplex_bench import runner as duplex_bench_runner

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOTA_PATH = Path(__file__).resolve().parent / "sota_reference.json"


def _load_sota() -> dict:
    return json.loads(SOTA_PATH.read_text(encoding="utf-8"))


def _compare_latency(measured_p50_ms: float | None, sota: dict) -> dict:
    budget = sota.get("latency_ms", {})
    lo = budget.get("target_cascaded_budget_p50_min", {}).get("value")
    hi = budget.get("target_cascaded_budget_p50_max", {}).get("value")
    if measured_p50_ms is None or lo is None or hi is None:
        return {"status": "no_data"}
    return {
        "status": "within_target" if lo <= measured_p50_ms <= hi else "outside_target",
        "measured_p50_ms": measured_p50_ms,
        "target_range_ms": [lo, hi],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-asr", action="store_true", help="Đo WER (gọi STT_BACKEND thật, tốn phí/GPU)")
    parser.add_argument(
        "--with-tool-call", action="store_true", help="Đo tool-call accuracy (gọi LLM_BACKEND thật, tốn phí)"
    )
    args = parser.parse_args()

    sota = _load_sota()
    entry: dict = {"ts": datetime.now(timezone.utc).isoformat()}

    print("=== Duplex-bench (Vietnamese turn-taking classifier) ===")
    duplex_summary = await duplex_bench_runner.run()
    entry["duplex_bench"] = {
        "accuracy": duplex_summary["accuracy"],
        "false_interrupt_rate": duplex_summary["false_interrupt_rate"],
        "missed_interrupt_rate": duplex_summary["missed_interrupt_rate"],
    }
    print(f"accuracy={duplex_summary['accuracy']:.2%}  n={duplex_summary['n']}")

    print("\n=== Latency (từ eval/results/latency_log.jsonl, hội thoại thật đã chạy) ===")
    latency_rows = latency_report.summarize()
    entry["latency"] = latency_rows
    ttfa_rows = [r for r in latency_rows if r["metric"] == "ttfa_ms"]
    ttfa_p50 = ttfa_rows[0]["p50_ms"] if ttfa_rows else None
    entry["latency_vs_target"] = _compare_latency(ttfa_p50, sota)
    if latency_rows:
        for row in latency_rows:
            print(
                f"{row['metric']:<16} {row['stt_backend']}/{row['llm_backend']}/{row['tts_backend']} "
                f"p50={row['p50_ms']:.0f}ms p95={row['p95_ms']:.0f}ms n={row['n']}"
            )
    else:
        print("Chưa có log latency (cần chạy hội thoại voice thật qua gateway/main.py trước).")

    if args.with_asr:
        print("\n=== ASR WER (gọi STT_BACKEND thật) ===")
        from eval import asr_wer

        asr_summary = await asr_wer.run()
        entry["asr_wer"] = {
            "backend": asr_summary["backend"],
            "overall_wer": asr_summary["overall_wer"],
            "by_domain_wer": asr_summary["by_domain_wer"],
            "by_noise_wer": asr_summary["by_noise_wer"],
            "code_switch_wer": asr_summary["code_switch_wer"],
        }
        ref = sota.get("asr_wer_pct", {}).get("zipformer_30m_rnnt_vlsp2020", {})
        entry["asr_wer_vs_sota"] = (
            {"status": "no_reference", "note": ref.get("source")}
            if ref.get("value") is None
            else {
                "status": "better" if asr_summary["overall_wer"] * 100 <= ref["value"] else "worse",
                "measured_pct": asr_summary["overall_wer"] * 100,
                "sota_pct": ref["value"],
            }
        )
        print(f"overall WER={asr_summary['overall_wer']:.3f}  backend={asr_summary['backend']}")
    else:
        print("\n(bỏ qua ASR WER — thêm --with-asr để đo, sẽ gọi STT_BACKEND thật)")

    if args.with_tool_call:
        print("\n=== Tool-call accuracy (gọi LLM_BACKEND thật) ===")
        from eval import tool_call_accuracy

        rows = []
        for case in tool_call_accuracy.TEST_CASES:
            rows.append(await tool_call_accuracy.run_case(case))
        accuracy = sum(r["correct"] for r in rows) / len(rows)
        entry["tool_call_accuracy"] = {"accuracy": accuracy, "n": len(rows)}
        print(f"accuracy={accuracy:.2%} (n={len(rows)})")
    else:
        print("\n(bỏ qua tool-call accuracy — thêm --with-tool-call để đo, sẽ gọi LLM_BACKEND thật)")

    benchmark_history.append(entry)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"benchmark_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi lịch sử: {benchmark_history.HISTORY_PATH}")
    print(f"Đã ghi chi tiết lần chạy: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
