"""Pipecat TTSService gọi selfhost/tts_server.py qua HTTP nội bộ (loopback) — chạy
trong .venv chính của dự án (không cần dependency nặng của f5-tts ở đây, chỉ cần
aiohttp). Xem selfhost/tts_server.py để biết vì sao tách venv riêng và cách chạy server.

F5-TTS sinh trọn câu bằng flow-matching (nhiều bước khử nhiễu) rồi mới trả về — KHÔNG
stream từng phần trong lúc tổng hợp như ElevenLabs/CosyVoice. Việc đọc response theo
chunk bên dưới chỉ là chia nhỏ để phát dần audio đã tổng hợp xong, không phải streaming
synthesis thật — cùng một điểm cần lưu ý như ZipformerVietnameseSTTService (xem
selfhost/asr.py).
"""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.tts_service import TTSService


class F5TTSVietnameseService(TTSService):
    """TTS tiếng Việt tự host, gọi selfhost/tts_server.py (F5-TTS-Vietnamese-ViVoice)."""

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: Optional[int] = 24000,
        **kwargs,
    ):
        super().__init__(push_stop_frames=True, sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            async with self._session.post(
                f"{self._base_url}/synthesize", json={"text": text}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(error=f"F5-TTS server error ({response.status}): {error}")
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"F5-TTS client error: {e}")
        finally:
            await self.stop_ttfb_metrics()
