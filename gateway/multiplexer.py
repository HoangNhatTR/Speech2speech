"""Stream Multiplexer + Traffic Scheduler — điều hướng message theo "modality" sang
đúng subject trên Event Bus, có giới hạn số request đồng thời (per-session + toàn cục)
trước khi publish. Bản tối giản: bảo vệ máy dev khỏi bị quá tải, không phải load
balancing thật (việc đó cần nhiều máy/worker)."""

import asyncio

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

_SESSION_MAX_CONCURRENT = 2
_GLOBAL_MAX_CONCURRENT = 8

_global_semaphore = asyncio.Semaphore(_GLOBAL_MAX_CONCURRENT)
_session_semaphores: dict[str, asyncio.Semaphore] = {}


def _session_semaphore(session_id: str) -> asyncio.Semaphore:
    return _session_semaphores.setdefault(
        session_id, asyncio.Semaphore(_SESSION_MAX_CONCURRENT)
    )


async def route(session_id: str, message: dict) -> dict:
    modality = message.get("modality")
    subject = MODALITY_SUBJECTS.get(modality)
    if subject is None:
        return {"error": f"unknown modality: {modality!r}"}

    payload = {k: v for k, v in message.items() if k != "modality"}
    payload["session_id"] = session_id

    async with _global_semaphore, _session_semaphore(session_id):
        return await bus_request(subject, payload)
