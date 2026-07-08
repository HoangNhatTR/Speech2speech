"""Speech2Speech — Pipeline giọng nói, hỗ trợ cả Giai đoạn 0 (API thương mại) và
Giai đoạn 1 (tự host tiếng Việt), chuyển đổi từng khối qua biến môi trường
STT_BACKEND/LLM_BACKEND/TTS_BACKEND=cloud|local (mặc định cloud). Đúng tinh thần
roadmap: "mỗi khối thay được" — phần còn lại của pipeline (VAD, tool calling, eval
harness, Event Bus) không đổi khi swap backend.

  cloud (Giai đoạn 0): WebRTC mic -> Deepgram STT -> Claude -> ElevenLabs TTS -> loa.
  local (Giai đoạn 1): WebRTC mic -> Zipformer STT (sherpa-onnx, tự host) -> Qwen3 qua
                        vLLM (tự host) -> F5-TTS-Vietnamese (tự host) -> loa.

Chạy: python -m gateway.main, rồi mở http://localhost:7860/client trong trình duyệt
(cần cấp quyền mic). Xem docstring trong gateway/main.py để biết vì sao không dùng
thẳng `python bot.py -t webrtc` (lệnh chuẩn của Pipecat) trên Python 3.10.

Speech Service (pipeline này) chạy trong-tiến-trình ở Gateway để giữ latency thấp cho
audio (xem docs/platform-architecture.md), nhưng vẫn tham gia Event Bus như mọi service
khác: khi người dùng hỏi giờ, LLM gọi tool "get_current_time" và pipeline này publish
request lên subject "svc.tool" cho Runtime Scheduler xử lý, giống hệt cách Gateway gọi
các service khác qua /v1/ws.

Backend "local" CHƯA chạy thử được đầy đủ trên máy dev (GPU 2GB, không đủ cho LLM/TTS
tự host) — xem docs/platform-architecture.md mục "Giai đoạn 1" để biết phần nào đã
verify (ASR) và phần nào mới chỉ viết code chờ máy có GPU đủ mạnh (LLM qua vLLM, TTS
F5-TTS-Vietnamese).

Yêu cầu (.env, xem .env.example): với backend cloud cần ANTHROPIC_API_KEY,
DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID. Cần NATS + runtime.dispatcher
đang chạy để tool calling hoạt động (xem README.md).
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams

from eval import latency_log
from eventbus.client import request as bus_request
from gateway.session_manager import session_manager
from selfhost.asr import ZipformerVietnameseSTTService
from selfhost.f5_tts_client import F5TTSVietnameseService

load_dotenv(override=True)

SYSTEM_PROMPT = """Bạn là một trợ lý giọng nói tiếng Việt, nói chuyện tự nhiên và ấm áp.
Đây là hội thoại BẰNG GIỌNG NÓI (được đọc thành tiếng), không phải văn bản, nên:
- Trả lời ngắn gọn, 1-3 câu mỗi lượt, như đang nói chuyện thật.
- Không dùng danh sách gạch đầu dòng, markdown, hay ký hiệu đặc biệt.
- Không đọc số liệu/URL dài dòng; diễn đạt tự nhiên như người thật.
- Nếu bị người dùng ngắt lời, hãy dừng ngay và lắng nghe.
- Khi cuộc trò chuyện vừa bắt đầu (chưa có câu nói nào của người dùng), hãy chủ động
  chào một câu ngắn gọn trước.
- Nếu người dùng hỏi giờ hiện tại, hãy gọi tool get_current_time thay vì tự đoán."""

GET_CURRENT_TIME_TOOL = FunctionSchema(
    name="get_current_time",
    description="Lấy giờ UTC hiện tại. Dùng khi người dùng hỏi mấy giờ rồi.",
    properties={},
    required=[],
)


async def handle_get_current_time(params: FunctionCallParams) -> None:
    """Gọi Tool Service qua Event Bus (svc.tool) thay vì xử lý tại chỗ — chứng minh
    Speech Service dùng chung hạ tầng với các service khác đứng sau Runtime Scheduler."""
    result = await bus_request("svc.tool", {"name": "get_current_time", "arguments": {}})
    await params.result_callback(result)


def webrtc_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


def build_stt():
    backend = os.getenv("STT_BACKEND", "cloud")
    if backend == "local":
        # Zipformer-30M-RNNT qua sherpa-onnx — đã verify chạy tốt trên CPU (xem
        # selfhost/asr.py). Tải model trước: python scripts/download_asr_model.py
        return ZipformerVietnameseSTTService(
            model_dir=os.environ["ASR_MODEL_DIR"],
        )
    return DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language=os.getenv("DEEPGRAM_LANGUAGE", "vi"),
            smart_format=True,
            interim_results=True,
            punctuate=True,
        ),
    )


def build_llm():
    backend = os.getenv("LLM_BACKEND", "cloud")
    if backend == "local":
        # Qwen3 qua vLLM (API tương thích OpenAI) — CHƯA chạy thử được: vLLM không có
        # wheel Windows (chỉ manylinux), cần chạy trong WSL2 hoặc GPU cloud Linux, và
        # máy dev (GPU 2GB) không đủ VRAM cho model 8B dù đã lượng tử hoá. Xem
        # docs/platform-architecture.md để biết lệnh `vllm serve` khi có máy phù hợp.
        llm = OpenAILLMService(
            base_url=os.environ["VLLM_BASE_URL"],
            api_key=os.getenv("VLLM_API_KEY", "not-needed"),
            settings=OpenAILLMService.Settings(
                model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-Instruct"),
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    else:
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    llm.register_function("get_current_time", handle_get_current_time)
    return llm


def build_tts(aiohttp_session: aiohttp.ClientSession):
    backend = os.getenv("TTS_BACKEND", "cloud")
    if backend == "local":
        # F5-TTS-Vietnamese-ViVoice qua selfhost/tts_server.py (venv riêng .venv-tts).
        # CHƯA chạy thử được: checkpoint ~5GB, máy dev chỉ có CPU/GPU 2GB — suy luận
        # flow-matching trên CPU sẽ rất chậm. Code đã viết đúng theo API thật của
        # f5_tts.api.F5TTS (xem selfhost/tts_server.py), cần máy có GPU đủ VRAM để dùng
        # thực tế. Chạy server trước: .venv-tts\\Scripts\\python selfhost/tts_server.py
        return F5TTSVietnameseService(
            base_url=os.getenv("F5_TTS_SERVER_URL", "http://localhost:8100"),
            aiohttp_session=aiohttp_session,
        )
    return ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            # eleven_flash_v2_5 là model duy nhất của ElevenLabs hỗ trợ tiếng Việt
            # với độ trễ thấp (~75ms), phù hợp mục tiêu latency của dự án này.
            model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
            language="vi",
        ),
    )


async def bot(runner_args: RunnerArguments):
    """Entry point required by the Pipecat dev runner."""
    transport = await create_transport(runner_args, {"webrtc": webrtc_params})

    aiohttp_session = aiohttp.ClientSession()
    stt = build_stt()
    llm = build_llm()
    tts = build_tts(aiohttp_session)

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(standard_tools=[GET_CURRENT_TIME_TOOL]),
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def _log_ttfa(observer, latency_seconds: float):
        value_ms = latency_seconds * 1000
        logger.info(f"[EVAL] TTFA (kết nối -> tiếng nói đầu tiên): {value_ms:.0f} ms")
        latency_log.append("ttfa_ms", value_ms)

    @latency_observer.event_handler("on_latency_measured")
    async def _log_turn_latency(observer, latency_seconds: float):
        value_ms = latency_seconds * 1000
        logger.info(f"[EVAL] Turn latency (user ngừng nói -> bot bắt đầu nói): {value_ms:.0f} ms")
        latency_log.append("turn_latency_ms", value_ms)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            TurnTrackingObserver(),
            TranscriptionLogObserver(),
            MetricsLogObserver(),
            latency_observer,
        ],
    )

    session = session_manager.create(kind="voice")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[gateway] Voice session {session.session_id} đã kết nối.")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[gateway] Voice session {session.session_id} đã ngắt kết nối.")
        session_manager.end(session.session_id)
        await aiohttp_session.close()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
