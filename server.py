"""Local dev server cho Speech2Speech (thay thế `pipecat.runner.run`).

Vì sao có file này: `pipecat.runner.run` (module "-t webrtc" chuẩn của Pipecat) import
`http.HTTPMethod`, vốn chỉ tồn tại từ Python 3.11 trở lên. Máy dev hiện dùng Python 3.10
(pipecat-ai bản mới nhất còn hỗ trợ 3.10 là 0.0.108; từ 1.0.0 trở đi Pipecat yêu cầu
Python >=3.11). File này dựng lại đúng phần cần thiết của dev runner (serve WebRTC
offer/answer + giao diện thử nghiệm trên trình duyệt) mà không cần `http.HTTPMethod`.

Nếu sau này nâng cấp lên Python 3.11+, có thể dùng lại `python bot.py -t webrtc` thay
cho file này.

Chạy: python server.py
Mở:  http://localhost:7860/client
"""

import asyncio

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from bot import bot
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


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connected(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection)
        background_tasks.add_task(bot, runner_args)

    return await webrtc_handler.handle_web_request(
        request=request, webrtc_connection_callback=on_connected
    )


@app.patch("/api/offer")
async def offer_ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


if __name__ == "__main__":
    logger.info(f"Bot ready! Mo http://{HOST}:{PORT}/client trong trinh duyet.")
    uvicorn.run(app, host=HOST, port=PORT)
