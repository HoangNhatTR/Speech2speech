"""Stream Multiplexer + Traffic Scheduler — điều hướng message theo "modality" sang
đúng subject trên Event Bus, có giới hạn số request đồng thời (per-session + toàn cục)
trước khi publish. Bản tối giản: bảo vệ máy dev khỏi bị quá tải, không phải load
balancing thật (việc đó cần nhiều máy/worker)."""

import asyncio
import os

from eventbus.client import request as bus_request

MODALITY_SUBJECTS = {
    "text": "svc.text",
    "vision": "svc.vision",
    "tool": "svc.tool",
    "memory": "svc.memory",
    "planning": "svc.planning",
    "reasoning": "svc.reasoning",
    "generation": "svc.generation",
}

def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


_SESSION_MAX_CONCURRENT = _positive_int_env("GATEWAY_SESSION_MAX_CONCURRENT", 2)
_GLOBAL_MAX_CONCURRENT = _positive_int_env("GATEWAY_GLOBAL_MAX_CONCURRENT", 8)

_global_semaphore = asyncio.Semaphore(_GLOBAL_MAX_CONCURRENT)
_session_semaphores: dict[str, asyncio.Semaphore] = {}


def _session_semaphore(session_id: str) -> asyncio.Semaphore:
    return _session_semaphores.setdefault(
        session_id, asyncio.Semaphore(_SESSION_MAX_CONCURRENT)
    )


def drop_session(session_id: str) -> None:
    """Dọn semaphore của session khi kết thúc. Không gọi hàm này thì _session_semaphores
    phình vô hạn theo số session từng kết nối (rò bộ nhớ trong tiến trình Gateway chạy
    lâu dài) vì setdefault() ở trên không bao giờ tự xoá entry cũ."""
    _session_semaphores.pop(session_id, None)


async def route(session_id: str, message: dict) -> dict:
    modality = message.get("modality")
    subject = MODALITY_SUBJECTS.get(modality)
    if subject is None:
        return {"error": f"unknown modality: {modality!r}"}

    payload = {k: v for k, v in message.items() if k != "modality"}
    payload["session_id"] = session_id

    async with _global_semaphore, _session_semaphore(session_id):
        return await bus_request(subject, payload)
