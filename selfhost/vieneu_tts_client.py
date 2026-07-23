"""Pipecat TTSService gọi selfhost/tts_server.py (VieNeu-TTS) qua HTTP nội bộ (loopback)
— chạy trong .venv chính của dự án (chỉ cần aiohttp ở đây, không cần dependency của
`vieneu`). Xem selfhost/tts_server.py để biết cách chạy server.

Mặc định gọi `/synthesize/stream`: server dùng `Vieneu.infer_stream()` và trả raw PCM16
ngay từ chunk đầu tiên, nên đây là streaming synthesis thật. Có thể đặt
`VIENEU_STREAMING=false` để quay về endpoint WAV `/synthesize`; client cũng tự fallback
nếu nối tới một TTS server phiên bản cũ chưa có endpoint streaming.
"""

import time
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.tts_service import TTSService


class VieNeuTTSService(TTSService):
    """TTS tiếng Việt tự host, gọi selfhost/tts_server.py (VieNeu-TTS)."""

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        streaming: bool = True,
        sample_rate: Optional[int] = 48000,
        **kwargs,
    ):
        super().__init__(push_stop_frames=True, sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._streaming = streaming

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        started_at = time.perf_counter()
        try:
            endpoint = "/synthesize/stream" if self._streaming else "/synthesize"
            response = await self._session.post(
                f"{self._base_url}{endpoint}", json={"text": text}
            )

            # Tương thích với TTS server cũ trong lúc rolling update.
            if self._streaming and response.status == 404:
                response.release()
                endpoint = "/synthesize"
                response = await self._session.post(
                    f"{self._base_url}{endpoint}", json={"text": text}
                )

            async with response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(error=f"VieNeu-TTS server error ({response.status}): {error}")
                    return

                await self.start_tts_usage_metrics(text)
                is_stream = endpoint.endswith("/stream")
                source_sample_rate = None
                if is_stream:
                    try:
                        source_sample_rate = int(response.headers["X-Audio-Sample-Rate"])
                    except (KeyError, TypeError, ValueError):
                        yield ErrorFrame(error="VieNeu-TTS streaming response thiếu sample rate hợp lệ")
                        return

                first_audio = True
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=not is_stream,
                    in_sample_rate=source_sample_rate,
                    context_id=context_id,
                ):
                    if first_audio:
                        first_audio = False
                        logger.info(
                            f"[EVAL] VieNeu TTS first audio: "
                            f"{(time.perf_counter() - started_at) * 1000:.0f} ms"
                        )
                        await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"VieNeu-TTS client error: {e}")
        finally:
            await self.stop_ttfb_metrics()
