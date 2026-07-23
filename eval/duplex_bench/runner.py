"""Chạy Vietnamese Turn-Taking Classifier Bench (xem eval/duplex_bench/__init__.py về
phạm vi, eval/duplex_bench/scenarios.py về kịch bản). Không cần audio/API/GPU thật —
chỉ mô phỏng chuỗi frame ASR và lái thẳng qua
`duplex.turn_strategies.VietnameseBackchannelTurnStartStrategy` (class thật đang chạy
trong bot.py khi DUPLEX_BACKCHANNEL_FILTER=true), nên chạy được trong CI/dashboard mà
không tốn phí — hợp với quy ước "liên tục benchmark".

Chạy: python -m eval.duplex_bench.runner
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.turns.types import ProcessFrameResult

from duplex.turn_strategies import VietnameseBackchannelTurnStartStrategy
from eval.duplex_bench.scenarios import SCENARIOS, Scenario

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


async def run_scenario(scenario: Scenario) -> dict:
    strategy = VietnameseBackchannelTurnStartStrategy()
    turn_started_at_ms: list[float] = []

    @strategy.event_handler("on_user_turn_started")
    async def _on_turn_started(_strategy, _params):
        turn_started_at_ms.append(current_t_ms[0])

    current_t_ms = [scenario.steps[0].t_ms if scenario.steps else 0.0]

    if scenario.bot_speaking_before:
        await strategy.process_frame(BotStartedSpeakingFrame())

    for step in scenario.steps:
        current_t_ms[0] = step.t_ms
        if step.frame == "interim":
            frame = InterimTranscriptionFrame(text=step.text, user_id="user", timestamp=str(step.t_ms))
        else:
            frame = TranscriptionFrame(text=step.text, user_id="user", timestamp=str(step.t_ms))
        result = await strategy.process_frame(frame)
        if result == ProcessFrameResult.STOP:
            break

    predicted_turn_started = bool(turn_started_at_ms)
    correct = predicted_turn_started == scenario.expected_turn_started
    t0 = scenario.steps[0].t_ms if scenario.steps else 0.0
    decided_at = turn_started_at_ms[0] if turn_started_at_ms else (scenario.steps[-1].t_ms if scenario.steps else t0)

    return {
        "id": scenario.id,
        "category": scenario.category,
        "expected_turn_started": scenario.expected_turn_started,
        "predicted_turn_started": predicted_turn_started,
        "correct": correct,
        "decision_latency_ms": round(decided_at - t0, 1),
        "note": scenario.note,
    }


def _rate(rows: list[dict], category: str, predicate) -> float | None:
    subset = [r for r in rows if r["category"] == category]
    if not subset:
        return None
    return sum(1 for r in subset if predicate(r)) / len(subset)


async def run() -> dict:
    """Chạy toàn bộ SCENARIOS, trả về summary dict (dùng lại được từ
    eval/run_benchmarks.py mà không cần parse file JSON)."""
    rows = [await run_scenario(s) for s in SCENARIOS]

    accuracy = sum(r["correct"] for r in rows) / len(rows)
    false_interrupt_rate = _rate(rows, "backchannel", lambda r: r["predicted_turn_started"])
    missed_interrupt_rate = _rate(rows, "real_interrupt", lambda r: not r["predicted_turn_started"])
    normal_turn_accuracy = _rate(rows, "normal_turn", lambda r: r["correct"])

    return {
        "n": len(rows),
        "accuracy": accuracy,
        "false_interrupt_rate": false_interrupt_rate,
        "missed_interrupt_rate": missed_interrupt_rate,
        "normal_turn_accuracy": normal_turn_accuracy,
        "rows": rows,
    }


async def main() -> None:
    summary = await run()

    for row in summary["rows"]:
        mark = "OK " if row["correct"] else "SAI"
        print(
            f"[{mark}] {row['id']:<16} cat={row['category']:<14} "
            f"expected={row['expected_turn_started']!s:<5} predicted={row['predicted_turn_started']!s:<5} "
            f"({row['decision_latency_ms']:.0f}ms)"
        )

    print(f"\nAccuracy tổng: {summary['accuracy']:.2%} (n={summary['n']})")
    print(f"False-interrupt rate (backchannel bị hiểu nhầm là ngắt lời): {summary['false_interrupt_rate']:.2%}")
    print(f"Missed-interrupt rate (ngắt lời thật bị bỏ sót): {summary['missed_interrupt_rate']:.2%}")
    print(f"Normal-turn accuracy (bot im lặng, mọi câu phải mở turn): {summary['normal_turn_accuracy']:.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"duplex_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
