"""Lightweight single-host dispatcher for small/medium local deployments.

It preserves the exact ``svc.<name>`` NATS contract used by ``runtime.dispatcher`` but
does not start Ray, Plasma object store, dashboard agents or worker processes. Each
service instance runs in this asyncio process. Use the Ray dispatcher only when the
services actually need multi-process/multi-node scheduling.

Run:
    python -m runtime.local_dispatcher
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from loguru import logger

from eventbus.client import close, subscribe
from services.generation_service import GenerationService
from services.memory_service import MemoryService
from services.planning_service import PlanningService
from services.reasoning_service import ReasoningService
from services.text_service import TextService
from services.tool_service import ToolService
from services.vision_service import VisionService

load_dotenv(override=True)

SERVICES = {
    "text": TextService,
    "vision": VisionService,
    "tool": ToolService,
    "memory": MemoryService,
    "planning": PlanningService,
    "reasoning": ReasoningService,
    "generation": GenerationService,
}


async def main() -> None:
    instances = {name: service_type() for name, service_type in SERVICES.items()}

    for name, service in instances.items():
        subject = f"svc.{name}"

        async def handler(request: dict, _service=service) -> dict:
            return await _service.handle(request)

        await subscribe(subject, handler)
        logger.info(f"Local Dispatcher: đang lắng nghe '{subject}' ({name})")

    logger.info(
        "Local Dispatcher sẵn sàng (single process, không Ray). Nhấn Ctrl+C để dừng."
    )
    try:
        await asyncio.Event().wait()
    finally:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
