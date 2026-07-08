"""Monitoring Service — subscriber thụ động, không nhận request/response như 7 service
kia. Lắng nghe mọi sự kiện đi qua Event Bus (subject "svc.>" và "session.>") và log lại.
Bản mỏng: chỉ log console; sau này có thể đẩy sang Prometheus/Grafana.

Chạy: python -m services.monitoring_service
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger

from eventbus.client import get_client

load_dotenv(override=True)


async def main() -> None:
    nc = await get_client()

    async def on_event(msg) -> None:
        logger.info(f"[MONITOR] {msg.subject}: {msg.data.decode(errors='replace')}")

    # NATS wildcard: "svc.>" khớp mọi subject con của "svc." (vd svc.text, svc.tool...).
    await nc.subscribe("svc.>", cb=on_event)
    await nc.subscribe("session.>", cb=on_event)

    logger.info("Monitoring Service: đang lắng nghe 'svc.>' và 'session.>'.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
