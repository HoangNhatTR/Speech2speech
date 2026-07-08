"""Gateway Layer — cổng vào duy nhất của hệ thống: Authentication, Session Manager,
Stream Multiplexer, Traffic Scheduler (4 mảng trong sơ đồ kiến trúc), trước khi giao
việc cho Event Bus / Runtime Scheduler. Thay thế `server.py` cũ (server.py chỉ có phần
WebRTC, chưa có auth/session/multiplex).

Vì sao dựng route WebRTC thủ công thay vì `pipecat.runner.run`: xem chú thích trong
lịch sử — `pipecat.runner.run` import `http.HTTPMethod`, chỉ có từ Python 3.11, còn máy
dev đang ở Python 3.10 (pipecat-ai bản mới nhất còn hỗ trợ 3.10 là 0.0.108).

Chạy: python -m gateway.main
  - Voice (Speech Service, WebRTC):     http://localhost:7860/client
  - Text/Vision/... (Event Bus, Ray):   ws://localhost:7860/v1/ws
"""

import json

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from bot import bot
from gateway.auth import is_authorized, require_auth
from gateway.multiplexer import route
from gateway.session_manager import session_manager
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
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

webrtc_handler = SmallWebRTCRequestHandler()


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer", dependencies=[Depends(require_auth)])
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connected(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
        background_tasks.add_task(bot, runner_args)

    return await webrtc_handler.handle_web_request(
        request=request, webrtc_connection_callback=on_connected
    )


@app.patch("/api/offer", dependencies=[Depends(require_auth)])
async def offer_ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


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
        logger.info(f"[gateway] WS session {session.session_id} đã ngắt kết nối.")


if __name__ == "__main__":
    logger.info(
        f"Gateway sẵn sàng. Voice: http://{HOST}:{PORT}/client — "
        f"Text/Vision/...: ws://{HOST}:{PORT}/v1/ws"
    )
    uvicorn.run(app, host=HOST, port=PORT)
