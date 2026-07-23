"""Speech2Speech — Pipeline giọng nói, hỗ trợ cả Giai đoạn 0 (API thương mại) và
Giai đoạn 1 (tự host tiếng Việt), chuyển đổi từng khối qua biến môi trường
STT_BACKEND/LLM_BACKEND/TTS_BACKEND=cloud|local (mặc định cloud). Đúng tinh thần
roadmap: "mỗi khối thay được" — phần còn lại của pipeline (VAD, tool calling, eval
harness, Event Bus) không đổi khi swap backend.

  cloud (Giai đoạn 0): WebRTC mic -> Deepgram STT -> Claude -> ElevenLabs TTS -> loa.
  local (Giai đoạn 1): WebRTC mic -> Zipformer STT (sherpa-onnx, tự host) -> Qwen3 qua
                        vLLM (tự host) -> VieNeu-TTS (tự host) -> loa.

Chạy: python -m gateway.main, rồi mở http://localhost:7860/client trong trình duyệt
(cần cấp quyền mic). Xem docstring trong gateway/main.py để biết vì sao không dùng
thẳng `python bot.py -t webrtc` (lệnh chuẩn của Pipecat) trên Python 3.10.

Speech Service (pipeline này) chạy trong-tiến-trình ở Gateway để giữ latency thấp cho
audio (xem docs/platform-architecture.md), nhưng vẫn tham gia Event Bus như mọi service
khác: khi người dùng hỏi giờ, LLM gọi tool "get_current_time" và pipeline này publish
request lên subject "svc.tool" cho Runtime Scheduler xử lý, giống hệt cách Gateway gọi
các service khác qua /v1/ws.

Backend "local": ASR (Zipformer), LLM (Qwen3-8B-AWQ qua vLLM) và TTS (VieNeu-TTS) ĐÃ
verify chạy thật — xem selfhost/asr.py, docs/platform-architecture.md, selfhost/
tts_server.py. vLLM cần venv riêng (.venv-vllm) và mất ~15 phút khởi động lần đầu (biên
dịch kernel CUDA cho GPU mới) — xem docs/platform-architecture.md mục "Giai đoạn 1" để
biết lệnh `vllm serve` đầy đủ và cách chạy.

Yêu cầu (.env, xem .env.example): với backend cloud cần ANTHROPIC_API_KEY,
DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID. Cần NATS + runtime.dispatcher
đang chạy để tool calling hoạt động (xem README.md).

Giai đoạn 2 (docs/roadmap.md mục 2, "Full duplex thật + cảm xúc") — CHƯA verify bằng hội
thoại thật, xem docstring từng file trong duplex/:
  DUPLEX_BACKCHANNEL_FILTER=true (mặc định) dùng duplex/turn_strategies.py để không cắt
  lời bot khi người dùng chỉ nói "dạ/vâng/ừm" (backchannel) trong lúc bot đang nói; đặt
  false để quay lại hành vi VAD thuần (ngắt ngay khi có bất kỳ tiếng nói nào).
  EMOTION_BACKEND=none (mặc định) | heuristic bật thử đường dây chèn tag cảm xúc vào
  context (duplex/emotion.py) — heuristic dựa trên từ khoá trong transcript, KHÔNG phải
  SER thật trên audio.
  duplex/interruption_recovery.py luôn bật: ghi lại vào context phần bot đã nói khi bị
  ngắt lời giữa chừng.
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
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from duplex.emotion import EmotionTaggingProcessor
from duplex.interruption_recovery import InterruptionRecoveryProcessor
from duplex.turn_strategies import VietnameseBackchannelTurnStartStrategy
from eval import latency_log
from eventbus.client import request as bus_request
from gateway import runtime_config
from gateway.session_manager import session_manager
from selfhost.asr import WhisperTurboVietnameseSTTService, ZipformerVietnameseSTTService
from selfhost.vieneu_tts_client import VieNeuTTSService
from s2s.audio_hub import AudioHub
from s2s.backend import build_s2s_backend
from s2s.events import S2SMode
from s2s.orchestrator import orchestrator_registry
from s2s.policy import SegmentPolicy
from s2s.processors import (
    AnchorObservationProcessor,
    AudioHubProcessor,
    PlaybackStateProcessor,
    RealtimeControlProcessor,
)
from s2s.shadow import ShadowS2SRunner

load_dotenv(override=True)

SYSTEM_PROMPT = """Bạn là một trợ lý giọng nói tiếng Việt, nói chuyện tự nhiên và ấm áp.
Đây là hội thoại BẰNG GIỌNG NÓI (được đọc thành tiếng), không phải văn bản, nên:
- Trả lời ngắn gọn, 1-3 câu mỗi lượt, như đang nói chuyện thật.
- Không dùng danh sách gạch đầu dòng, markdown, hay ký hiệu đặc biệt.
- Không đọc số liệu/URL dài dòng; diễn đạt tự nhiên như người thật.
- Nếu bị người dùng ngắt lời, hãy dừng ngay và lắng nghe.
- Khi cuộc trò chuyện vừa bắt đầu (chưa có câu nói nào của người dùng), hãy chủ động
  chào một câu ngắn gọn trước.
- Nếu người dùng hỏi giờ hiện tại, hãy gọi tool get_current_time thay vì tự đoán.
- Thỉnh thoảng bạn sẽ thấy các ghi chú dạng "[Hệ thống: ...]" hoặc "[user_emotion: ...]"
  xen trong hội thoại — đó là ghi chú nội bộ, KHÔNG phải lời người dùng nói. Dùng nó để
  điều chỉnh phản hồi (vd nói lại phần bị ngắt, hoặc dịu giọng nếu người dùng đang khó
  chịu), tuyệt đối không đọc nguyên văn ghi chú đó ra cho người dùng nghe."""

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
    backend = runtime_config.get("STT_BACKEND", "cloud")
    if backend == "local":
        # Hai engine local, chọn qua ASR_LOCAL_ENGINE (mặc định zipformer) — xem
        # selfhost/asr.py cho chi tiết đánh đổi giữa hai lựa chọn.
        engine = os.getenv("ASR_LOCAL_ENGINE", "zipformer")
        if engine == "whisper":
            # Whisper large-v3-turbo qua sherpa-onnx — ĐÃ ĐO, KHÔNG khuyến nghị: WER 12.5%
            # (thua Zipformer), latency ~2474ms/câu — xem selfhost/asr.py để biết chi tiết.
            return WhisperTurboVietnameseSTTService(
                model_dir=os.environ["WHISPER_ASR_MODEL_DIR"],
            )
        # Zipformer train ~70k giờ qua sherpa-onnx — đã verify chạy tốt trên CPU, WER 2.7%
        # (xem selfhost/asr.py). Tải model trước: python scripts/download_asr_model.py
        return ZipformerVietnameseSTTService(
            model_dir=os.environ["ASR_MODEL_DIR"],
        )
    return DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language=runtime_config.get("DEEPGRAM_LANGUAGE", "vi"),
            smart_format=True,
            interim_results=True,
            punctuate=True,
        ),
    )


def build_llm():
    backend = runtime_config.get("LLM_BACKEND", "cloud")
    if backend == "local":
        # Qwen3-8B-AWQ qua vLLM (API tương thích OpenAI) — ĐÃ verify chạy thật trên GPU
        # (venv riêng .venv-vllm, xem docs/platform-architecture.md). Dùng bản lượng tử
        # hoá AWQ (~6.1GB, không phải bf16 gốc ~16GB) vì máy chạy chung với người khác,
        # RAM/VRAM hạn chế theo thời điểm — đổi VLLM_MODEL nếu máy bạn dư tài nguyên hơn.
        # Cờ --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
        # BẮT BUỘC khi chạy `vllm serve` (xem lệnh đầy đủ trong platform-architecture.md)
        # — thiếu reasoning-parser thì nội dung suy luận `<think>...</think>` sẽ lẫn vào
        # content thật, TTS sẽ đọc to cả phần suy luận đó lên loa.
        llm = OpenAILLMService(
            base_url=os.environ["VLLM_BASE_URL"],
            api_key=os.getenv("VLLM_API_KEY", "not-needed"),
            settings=OpenAILLMService.Settings(
                model=runtime_config.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    else:
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=runtime_config.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    llm.register_function("get_current_time", handle_get_current_time)
    return llm


def build_tts(aiohttp_session: aiohttp.ClientSession):
    backend = runtime_config.get("TTS_BACKEND", "cloud")
    if backend == "local":
        # VieNeu-TTS qua selfhost/tts_server.py (venv riêng .venv-tts). ĐÃ verify chạy
        # thật, cả CPU (RTF ~1.1-1.3) lẫn GPU (RTF ~1.0) — xem selfhost/tts_server.py.
        # Chạy server trước: .venv-tts\\Scripts\\python selfhost/tts_server.py (Windows)
        # hoặc .venv-tts/bin/python selfhost/tts_server.py (Linux/macOS)
        return VieNeuTTSService(
            base_url=os.getenv("VIENEU_SERVER_URL", "http://localhost:8100"),
            aiohttp_session=aiohttp_session,
            streaming=runtime_config.get("VIENEU_STREAMING", "true").strip().lower()
            == "true",
        )
    return ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            # eleven_flash_v2_5 là model duy nhất của ElevenLabs hỗ trợ tiếng Việt
            # với độ trễ thấp (~75ms), phù hợp mục tiêu latency của dự án này.
            model=runtime_config.get("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
            language="vi",
        ),
    )


async def bot(runner_args: RunnerArguments):
    """Entry point required by the Pipecat dev runner."""
    transport = await create_transport(runner_args, {"webrtc": webrtc_params})

    aiohttp_session = aiohttp.ClientSession()
    session = session_manager.create(kind="voice")
    try:
        s2s_mode = S2SMode.parse(runtime_config.get("S2S_MODE", "shadow"))
    except ValueError:
        session_manager.end(session.session_id)
        await aiohttp_session.close()
        raise
    orchestrator = orchestrator_registry.create(session.session_id, s2s_mode)
    try:
        ring_ms = int(runtime_config.get("S2S_AUDIO_RING_MS", "30000"))
        subscriber_frames = int(
            runtime_config.get("S2S_SUBSCRIBER_QUEUE_FRAMES", "128")
        )
    except ValueError:
        logger.warning("S2S Audio Hub config không hợp lệ; dùng 30000ms/128 frames")
        ring_ms, subscriber_frames = 30_000, 128
    audio_hub = AudioHub(
        max_buffer_ms=ring_ms,
        subscriber_queue_frames=subscriber_frames,
    )
    segment_policy = SegmentPolicy()
    audio_hub_processor = AudioHubProcessor(audio_hub, orchestrator)
    realtime_control = RealtimeControlProcessor(orchestrator, segment_policy)
    anchor_observer = AnchorObservationProcessor(orchestrator, segment_policy)
    playback_state = PlaybackStateProcessor(orchestrator)

    shadow_runner = None
    if s2s_mode is not S2SMode.OFF:
        shadow_backend = build_s2s_backend(
            runtime_config.get("S2S_SHADOW_BACKEND", "probe")
        )
        shadow_runner = ShadowS2SRunner(
            subscription=audio_hub.subscribe("speech-native-fast-path"),
            backend=shadow_backend,
            orchestrator=orchestrator,
            on_proposal=realtime_control.submit_fast_proposal,
        )

    stt = build_stt()
    llm = build_llm()
    tts = build_tts(aiohttp_session)

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(standard_tools=[GET_CURRENT_TIME_TOOL]),
    )

    # Giai đoạn 2 — xem duplex/turn_strategies.py: mặc định thay VADUserTurnStartStrategy
    # (ngắt ngay khi có tiếng động) bằng bản phân biệt backchannel tiếng Việt. Đặt
    # DUPLEX_BACKCHANNEL_FILTER=false để quay lại hành vi VAD thuần nếu cần so sánh.
    backchannel_filter = runtime_config.get("DUPLEX_BACKCHANNEL_FILTER", "true").strip().lower() == "true"
    try:
        barge_in_delay_ms = int(runtime_config.get("DUPLEX_BARGE_IN_DELAY_MS", "200"))
    except ValueError:
        logger.warning("DUPLEX_BARGE_IN_DELAY_MS không hợp lệ; dùng 200ms")
        barge_in_delay_ms = 200
    user_turn_strategies = None
    if backchannel_filter:
        user_turn_strategies = UserTurnStrategies(
            start=[
                VietnameseBackchannelTurnStartStrategy(
                    barge_in_delay_ms=barge_in_delay_ms,
                )
            ]
        )

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=user_turn_strategies,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            audio_hub_processor,
            stt,
            EmotionTaggingProcessor(context),
            context_aggregator.user(),
            realtime_control,
            llm,
            anchor_observer,
            tts,
            playback_state,
            InterruptionRecoveryProcessor(context),
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

    cleaned_up = False

    async def cleanup(*, cancel_task: bool) -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        if shadow_runner:
            await shadow_runner.stop()
        stats = audio_hub.stats()
        logger.info(
            f"[s2s] Session {session.session_id}: mode={s2s_mode.value}, "
            f"audio_frames={stats.published_frames}, dropped={stats.dropped_subscriber_frames}"
        )
        audio_hub.close()
        orchestrator_registry.remove(session.session_id)
        session_manager.end(session.session_id)
        if not aiohttp_session.closed:
            await aiohttp_session.close()
        if cancel_task:
            await task.cancel()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(
            f"[gateway] Voice session {session.session_id} đã kết nối "
            f"(S2S_MODE={s2s_mode.value})."
        )
        if shadow_runner:
            await shadow_runner.start()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[gateway] Voice session {session.session_id} đã ngắt kết nối.")
        await cleanup(cancel_task=True)

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.run(task)
    finally:
        await cleanup(cancel_task=False)
