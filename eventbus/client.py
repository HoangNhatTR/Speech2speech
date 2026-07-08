"""Event Bus — wrapper mỏng quanh NATS (nats-py).

Đây là điểm swap-out duy nhất nếu sau này đổi backend (Kafka/Redis Streams): chỉ cần
viết lại các hàm publish/request/subscribe bên dưới, phần còn lại của hệ thống
(gateway/, runtime/, services/, bot.py) không cần đổi gì.

Quy ước subject: mỗi service request/response đứng sau Runtime Scheduler dùng
"svc.<tên>" (vd: "svc.text", "svc.tool"). Sự kiện fire-and-forget (không đợi phản hồi,
dùng để Monitoring Service nghe) dùng tiền tố khác, vd "session.started".
"""

import json
import os
from typing import Awaitable, Callable

import nats
from loguru import logger
from nats.aio.client import Client as NATSClient
from nats.aio.msg import Msg

_client: NATSClient | None = None

Handler = Callable[[dict], Awaitable[dict | None]]


async def get_client() -> NATSClient:
    global _client
    if _client is None or not _client.is_connected:
        url = os.getenv("NATS_URL", "nats://localhost:4222")
        _client = await nats.connect(url)
    return _client


async def publish(subject: str, payload: dict) -> None:
    """Gửi sự kiện, không đợi phản hồi (fire-and-forget)."""
    nc = await get_client()
    await nc.publish(subject, json.dumps(payload).encode())


async def request(subject: str, payload: dict, timeout: float = 10.0) -> dict:
    """Gửi request và đợi phản hồi (dùng để gọi một service đứng sau Runtime Scheduler)."""
    nc = await get_client()
    msg = await nc.request(subject, json.dumps(payload).encode(), timeout=timeout)
    return json.loads(msg.data.decode())


async def subscribe(subject: str, handler: Handler) -> None:
    """Lắng nghe `subject`. Nếu `handler` trả về dict và message có reply-subject
    (tức được gọi qua `request()`), tự động gửi lại kết quả."""
    nc = await get_client()

    async def _on_msg(msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode()) if msg.data else {}
        except json.JSONDecodeError:
            data = {"_raw": msg.data.decode(errors="replace")}

        try:
            result = await handler(data)
        except Exception as exc:  # noqa: BLE001 — lỗi service không được để request treo tới timeout
            logger.exception(f"Loi khi xu ly message tren subject '{subject}'")
            if msg.reply:
                await msg.respond(json.dumps({"error": str(exc)}).encode())
            return

        if msg.reply and result is not None:
            await msg.respond(json.dumps(result).encode())

    await nc.subscribe(subject, cb=_on_msg)


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
