"""Authentication — API key tĩnh, tối giản cho 1 dev. Đặt GATEWAY_AUTH_DISABLED=true
(mặc định) để bỏ qua kiểm tra khi dev local — cần vì giao diện thử nghiệm WebRTC có sẵn
của Pipecat không gửi header tuỳ chỉnh. Trước khi mở ra ngoài localhost: đặt
GATEWAY_AUTH_DISABLED=false và điền GATEWAY_API_KEYS."""

import os

from fastapi import HTTPException, Request


def auth_disabled() -> bool:
    return os.getenv("GATEWAY_AUTH_DISABLED", "true").strip().lower() == "true"


def valid_keys() -> set[str]:
    raw = os.getenv("GATEWAY_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def is_authorized(token: str | None) -> bool:
    return auth_disabled() or (token is not None and token in valid_keys())


async def require_auth(request: Request) -> None:
    """Dependency cho route HTTP: đọc header `Authorization: Bearer <key>`."""
    header = request.headers.get("authorization", "")
    token = header[len("Bearer ") :] if header.startswith("Bearer ") else None
    if not is_authorized(token):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
