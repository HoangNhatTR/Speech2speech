"""Gateway Layer — cổng vào duy nhất của hệ thống: Authentication, Session Manager,
Stream Multiplexer, Traffic Scheduler (4 mảng trong sơ đồ kiến trúc), trước khi giao
việc cho Event Bus / Runtime Scheduler. Thay thế `server.py` cũ (server.py chỉ có phần
WebRTC, chưa có auth/session/multiplex).

Vì sao dựng route WebRTC thủ công thay vì `pipecat.runner.run`: lý do gốc (import
`http.HTTPMethod`, chỉ có từ Python 3.11) **không còn đúng trên máy này** — `.venv` thực
tế chạy Python 3.12.3 (xác nhận bằng `python --version`), import đó chạy bình thường.
Vẫn giữ route thủ công vì đã có auth/session_manager/multiplexer riêng gắn vào, không
muốn thay bằng toàn bộ `pipecat.runner.run` (còn kèm route WhatsApp/telephony không cần).

NHƯNG việc hand-roll trước đây bỏ sót một điểm quan trọng: giao diện WebRTC dựng sẵn
(`pipecat_ai_small_webrtc_prebuilt`) có JS đã build cứng gọi `POST /start` trước
(`startBotParams: {endpoint: "/start", ...}` — xác nhận bằng cách grep thẳng vào file
JS đã build, không đoán), rồi mới gọi `/sessions/{sessionId}/api/offer` cho SDP thật —
KHÔNG gọi thẳng `/api/offer` như route cũ ở đây cung cấp. Thiếu `/start` khiến UI báo
"not connected to agent" (404 khi POST /start). Đã thêm 2 route bên dưới mô phỏng đúng
theo `pipecat.runner.run._setup_small_webrtc_routes` (đọc source thật trong
`pipecat/runner/run.py` của venv, không đoán field name) để khớp đúng hợp đồng UI cần.

Chạy: python -m gateway.main
  - Voice (Speech Service, WebRTC):     http://localhost:7860/client
  - Text/Vision/... (Event Bus, Ray):   ws://localhost:7860/v1/ws
  - Dashboard/Settings API (FE riêng, xem frontend/): http://localhost:7860/api/*
    (gateway/dashboard_api.py — status/config/metrics/benchmarks)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from bot import bot
from gateway.auth import is_authorized, require_auth
from gateway.dashboard_api import router as dashboard_router
from gateway.multiplexer import drop_session, route
from gateway.session_manager import session_manager
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

HOST = "localhost"
PORT = 7860

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/client", SmallWebRTCPrebuiltUI)
app.include_router(dashboard_router)

webrtc_handler = SmallWebRTCRequestHandler()
_active_voice_runs: set[str] = set()


def _voice_max_sessions() -> int:
    try:
        value = int(os.getenv("VOICE_MAX_SESSIONS", "2"))
    except ValueError:
        return 2
    return max(1, value)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer", dependencies=[Depends(require_auth)])
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connected(connection: SmallWebRTCConnection):
        if len(_active_voice_runs) >= _voice_max_sessions():
            logger.warning(
                "Từ chối voice session mới: đã đạt VOICE_MAX_SESSIONS={}.",
                _voice_max_sessions(),
            )
            await connection.disconnect()
            return

        run_id = uuid.uuid4().hex
        _active_voice_runs.add(run_id)
        runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)

        async def run_voice_bot() -> None:
            try:
                await bot(runner_args)
            finally:
                _active_voice_runs.discard(run_id)

        background_tasks.add_task(run_voice_bot)

    return await webrtc_handler.handle_web_request(
        request=request, webrtc_connection_callback=on_connected
    )


@app.patch("/api/offer", dependencies=[Depends(require_auth)])
async def offer_ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


class IceServer(TypedDict, total=False):
    urls: Union[str, List[str]]


class IceConfig(TypedDict):
    iceServers: List[IceServer]


class StartBotResult(TypedDict, total=False):
    sessionId: str
    iceConfig: Optional[IceConfig]


# In-memory, một tiến trình — giống session_manager.py. session_id ở đây chỉ là "vé" để
# UI dựng sẵn nối /start với lần gọi /sessions/{id}/api/offer ngay sau đó (vài trăm ms),
# không phải phiên hội thoại lâu dài (đó là việc của session_manager) nên không cần dọn
# định kỳ như multiplexer.py phải làm với _session_semaphores.
_pending_sessions: Dict[str, Dict[str, Any]] = {}


@app.post("/start", dependencies=[Depends(require_auth)])
async def start_session(request: Request) -> StartBotResult:
    """Điểm vào ĐẦU TIÊN mà giao diện WebRTC dựng sẵn của Pipecat gọi khi bấm Connect
    (bundle JS cố định `endpoint: "/start"`, không cấu hình được — xem docstring đầu
    file). Mô phỏng `/start` của Pipecat Cloud: phát session_id, UI dùng nó để gọi
    `/sessions/{session_id}/api/offer` ngay sau — xem `session_proxy` bên dưới."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    _pending_sessions[session_id] = request_data.get("requestData") or request_data.get("request_data") or {}

    result: StartBotResult = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = IceConfig(iceServers=[IceServer(urls=["stun:stun.l.google.com:19302"])])
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def session_proxy(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """Chuyển tiếp SDP offer/ICE candidate thật (UI gọi tới đây, không phải thẳng
    `/api/offer`) sang đúng handler đã có ở trên. Không check `require_auth` riêng ở đây
    — giống cách `pipecat.runner.run` tự làm — vì session_id là UUID ngẫu nhiên chỉ phát
    được qua `/start` (đã có `require_auth`), không phải một cổng vào độc lập."""
    if session_id not in _pending_sessions:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        try:
            request_data = await request.json()
        except Exception as e:
            return Response(content=f"Invalid WebRTC request: {e}", status_code=400)

        if request.method == "POST":
            webrtc_request = SmallWebRTCRequest(
                sdp=request_data["sdp"],
                type=request_data["type"],
                pc_id=request_data.get("pc_id"),
                restart_pc=request_data.get("restart_pc"),
                request_data=request_data.get("request_data")
                or request_data.get("requestData")
                or _pending_sessions[session_id],
            )
            return await offer(webrtc_request, background_tasks)
        elif request.method == "PATCH":
            patch_request = SmallWebRTCPatchRequest(
                pc_id=request_data["pc_id"],
                candidates=[IceCandidate(**c) for c in request_data.get("candidates", [])],
            )
            return await offer_ice_candidate(patch_request)

    return Response(status_code=200)


@app.websocket("/v1/ws")
async def ws_endpoint(websocket: WebSocket):
    """Client SDK cho các modality không phải audio: gửi
    `{"modality": "text"|"vision"|"tool"|"memory"|"planning"|"reasoning"|"generation", ...}`,
    Stream Multiplexer điều hướng qua Event Bus tới đúng service, trả kết quả qua cùng kết nối.
    Auth qua query param (WebSocket API của trình duyệt không set header tuỳ ý được):
    `ws://localhost:7860/v1/ws?token=<GATEWAY_API_KEYS>`.
    """
    token = websocket.query_params.get("token")
    if not is_authorized(token):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    session = session_manager.create(kind="ws")
    logger.info(f"[gateway] WS session {session.session_id} đã kết nối.")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue
            result = await route(session.session_id, message)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        session_manager.end(session.session_id)
        drop_session(session.session_id)
        logger.info(f"[gateway] WS session {session.session_id} đã ngắt kết nối.")


if __name__ == "__main__":
    logger.info(
        f"Gateway sẵn sàng. Voice: http://{HOST}:{PORT}/client — "
        f"Text/Vision/...: ws://{HOST}:{PORT}/v1/ws"
    )
    uvicorn.run(app, host=HOST, port=PORT)
