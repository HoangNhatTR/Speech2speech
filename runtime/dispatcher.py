"""Runtime Scheduler — khởi tạo Ray (local mode) với 1 actor cho mỗi service đứng sau
nó, subscribe subject "svc.<tên>" trên Event Bus, và forward request vào actor tương
ứng. Speech Service không đứng sau đây (chạy trong-tiến-trình ở Gateway vì lý do
latency); Monitoring Service cũng không (nó là subscriber thụ động riêng). Xem
docs/platform-architecture.md.

Chạy: python -m runtime.dispatcher
"""

import asyncio
import os

import ray
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

# Ray worker (mỗi actor chạy trong 1 process riêng) không tự kế thừa os.environ đã
# được load_dotenv() nạp ở process driver — phải truyền tường minh qua runtime_env,
# nếu không các service gọi Claude sẽ báo thiếu ANTHROPIC_API_KEY dù .env đã có key.
ACTOR_ENV_VARS = {
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", ""),
}


async def main() -> None:
    ray.init(
        include_dashboard=False,
        log_to_driver=True,
        runtime_env={"env_vars": ACTOR_ENV_VARS},
    )

    for name, cls in SERVICES.items():
        actor = ray.remote(cls).remote()
        subject = f"svc.{name}"

        async def handler(request: dict, _actor=actor) -> dict:
            return await _actor.handle.remote(request)

        await subscribe(subject, handler)
        logger.info(f"Runtime Scheduler: đang lắng nghe '{subject}' ({name})")

    logger.info("Runtime Scheduler sẵn sàng. Nhấn Ctrl+C để dừng.")
    try:
        await asyncio.Event().wait()
    finally:
        await close()
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
