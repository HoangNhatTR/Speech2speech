"""Speech2Speech — Giai đoạn 0: khung xương (Pipecat + WebRTC + API thương mại).

Pipeline: WebRTC mic -> Deepgram STT -> Claude (Anthropic) -> ElevenLabs TTS -> WebRTC loa.
Mục tiêu giai đoạn này KHÔNG phải chất lượng cuối cùng, mà là có một hệ thống streaming
chạy được để:
  - chốt UX (interruption/barge-in, turn-taking) bằng framework có sẵn (Pipecat),
  - đo latency từng chặng (xem log "[EVAL]"),
  - làm eval harness trước khi thay từng khối bằng model tự host (giai đoạn 1+).

Chạy: python server.py, rồi mở http://localhost:7860/client trong trình duyệt (cần cấp
quyền mic). Xem docstring trong server.py để biết vì sao không dùng thẳng
`python bot.py -t webrtc` (lệnh chuẩn của Pipecat) trên Python 3.10.

Yêu cầu: điền ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY,
ELEVENLABS_VOICE_ID vào file .env (xem .env.example).
"""

import os

from dotenv import load_dotenv
from loguru import logger

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
from pipecat.transports.base_transport import TransportParams

load_dotenv(override=True)

SYSTEM_PROMPT = """Bạn là một trợ lý giọng nói tiếng Việt, nói chuyện tự nhiên và ấm áp.
Đây là hội thoại BẰNG GIỌNG NÓI (được đọc thành tiếng), không phải văn bản, nên:
- Trả lời ngắn gọn, 1-3 câu mỗi lượt, như đang nói chuyện thật.
- Không dùng danh sách gạch đầu dòng, markdown, hay ký hiệu đặc biệt.
- Không đọc số liệu/URL dài dòng; diễn đạt tự nhiên như người thật.
- Nếu bị người dùng ngắt lời, hãy dừng ngay và lắng nghe.
- Khi cuộc trò chuyện vừa bắt đầu (chưa có câu nói nào của người dùng), hãy chủ động
  chào một câu ngắn gọn trước."""


def webrtc_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


async def bot(runner_args: RunnerArguments):
    """Entry point required by the Pipecat dev runner."""
    transport = await create_transport(runner_args, {"webrtc": webrtc_params})

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language=os.getenv("DEEPGRAM_LANGUAGE", "vi"),
            smart_format=True,
            interim_results=True,
            punctuate=True,
        ),
    )

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            # eleven_flash_v2_5 là model duy nhất của ElevenLabs hỗ trợ tiếng Việt
            # với độ trễ thấp (~75ms), phù hợp mục tiêu latency của dự án này.
            model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
            language="vi",
        ),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
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
        logger.info(f"[EVAL] TTFA (kết nối -> tiếng nói đầu tiên): {latency_seconds * 1000:.0f} ms")

    @latency_observer.event_handler("on_latency_measured")
    async def _log_turn_latency(observer, latency_seconds: float):
        logger.info(
            f"[EVAL] Turn latency (user ngừng nói -> bot bắt đầu nói): {latency_seconds * 1000:.0f} ms"
        )

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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client đã kết nối, bắt đầu hội thoại.")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client đã ngắt kết nối.")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
