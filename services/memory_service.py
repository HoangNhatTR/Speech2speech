"""Memory Service — lịch sử hội thoại theo session. Bản mỏng: dict in-memory bên trong
Ray actor (mỗi actor có state riêng, tồn tại suốt vòng đời tiến trình dispatcher).
Đổi sang Redis khi cần nhiều tiến trình/instance chia sẻ cùng state.

Request: {"action": "append"|"get_history"|"clear", "session_id": "...", "message": {...}}
"""

from services.base import BaseService


class MemoryService(BaseService):
    def __init__(self) -> None:
        self._history: dict[str, list[dict]] = {}

    async def handle(self, request: dict) -> dict:
        action = request.get("action")
        session_id = request.get("session_id", "default")

        if action == "append":
            self._history.setdefault(session_id, []).append(request["message"])
            return {"ok": True}
        if action == "get_history":
            return {"history": self._history.get(session_id, [])}
        if action == "clear":
            self._history.pop(session_id, None)
            return {"ok": True}
        return {"error": f"unknown action: {action}"}
