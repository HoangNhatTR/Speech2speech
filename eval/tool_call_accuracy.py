"""Đo tool-call accuracy của LLM_BACKEND đang cấu hình: với mỗi câu hỏi, model có gọi
đúng tool (hoặc đúng KHÔNG gọi tool nào) như kỳ vọng không. Dùng lại nguyên các thành
phần đã verify chạy được trong bot.py (LLMContext, ToolsSchema, register_function,
LLMContextAggregatorPair, PipelineTask) — chỉ bỏ transport/STT/TTS vì ở đây chỉ cần
test quyết định gọi tool của LLM, không cần audio.

CHƯA chạy được trong lần build này vì cần ANTHROPIC_API_KEY thật (hoặc VLLM_BASE_URL
thật nếu LLM_BACKEND=local) — xem docs/platform-architecture.md.

Chạy: python -m eval.tool_call_accuracy
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from bot import GET_CURRENT_TIME_TOOL, SYSTEM_PROMPT, build_llm
from pipecat.adapters.schemas.tools_schema import ToolsSchema

load_dotenv(override=True)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class TestCase:
    question: str
    expected_tool: str | None  # None nghĩa là KHÔNG nên gọi tool nào


TEST_CASES = [
    TestCase("Bây giờ là mấy giờ rồi?", "get_current_time"),
    TestCase("Cho tôi biết giờ UTC hiện tại đi.", "get_current_time"),
    TestCase("Hôm nay thời tiết thế nào?", None),
    TestCase("Bạn tên là gì?", None),
    TestCase("Kể cho tôi một câu chuyện cười.", None),
]


@dataclass
class Called:
    tool_names: list = field(default_factory=list)


class _TurnEndWatcher(FrameProcessor):
    """Set `done` ngay khi lượt LLM kết thúc, để không phải chờ hết timeout_s mỗi lần
    (turn không gọi tool vẫn kết thúc nhanh, không có lý do gì đợi thêm)."""

    def __init__(self, done: asyncio.Event, **kwargs):
        super().__init__(**kwargs)
        self._done = done

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._done.set()
        await self.push_frame(frame, direction)


async def run_case(case: TestCase, timeout_s: float = 15.0) -> dict:
    called = Called()
    done = asyncio.Event()

    async def handle_get_current_time(params: FunctionCallParams) -> None:
        called.tool_names.append(params.function_name)
        await params.result_callback({"utc_time": "2026-01-01T00:00:00Z"})
        done.set()

    llm = build_llm()
    # build_llm() đã register_function cho get_current_time trỏ vào Event Bus thật;
    # ở đây override lại bằng handler nội bộ để test không phụ thuộc NATS đang chạy.
    llm.register_function("get_current_time", handle_get_current_time)

    context = LLMContext(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.question},
        ],
        tools=ToolsSchema(standard_tools=[GET_CURRENT_TIME_TOOL]),
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [context_aggregator.user(), llm, _TurnEndWatcher(done), context_aggregator.assistant()]
    )
    task = PipelineTask(pipeline, params=PipelineParams())
    runner = PipelineRunner(handle_sigint=False)

    async def _drive():
        await task.queue_frames([LLMRunFrame()])
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        await task.cancel()

    try:
        await asyncio.wait_for(asyncio.gather(runner.run(task), _drive()), timeout=timeout_s + 5)
    except asyncio.TimeoutError:
        pass

    actual_tool = called.tool_names[0] if called.tool_names else None
    correct = actual_tool == case.expected_tool
    return {
        "question": case.question,
        "expected_tool": case.expected_tool,
        "actual_tool": actual_tool,
        "correct": correct,
    }


async def main() -> None:
    llm = build_llm()
    backend_name = type(llm).__name__
    print(f"LLM backend: {backend_name}\n")

    rows = []
    for case in TEST_CASES:
        row = await run_case(case)
        rows.append(row)
        mark = "OK" if row["correct"] else "SAI"
        print(f"[{mark}] {row['question']!r} -> expected={row['expected_tool']}, actual={row['actual_tool']}")

    accuracy = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nTool-call accuracy: {accuracy:.2%} ({sum(r['correct'] for r in rows)}/{len(rows)})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"tool_call_accuracy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(
        json.dumps({"backend": backend_name, "accuracy": accuracy, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã ghi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
