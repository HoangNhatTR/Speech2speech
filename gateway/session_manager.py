"""Session Manager — theo dõi các phiên đang hoạt động (in-memory, đủ cho 1 tiến trình
Gateway). Dùng chung cho cả phiên voice (WebRTC, tạo trong bot.py) và phiên text/vision
(WebSocket, tạo trong gateway/main.py) để có một nơi duy nhất biết ai đang kết nối.
Khi cần nhiều instance Gateway (nhiều máy), chuyển state này sang Redis."""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    kind: str
    created_at: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, kind: str) -> Session:
        session = Session(session_id=uuid.uuid4().hex, kind=kind)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        return len(self._sessions)

    def list_sessions(self) -> list[Session]:
        """Snapshot các phiên đang mở — dùng cho GET /api/status (Dashboard). Chỉ chứa
        session_id/kind/created_at, không có nội dung hội thoại/audio nào."""
        return list(self._sessions.values())


session_manager = SessionManager()
