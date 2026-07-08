"""ASR tự host cho tiếng Việt — Zipformer-30M-RNNT (hynt/Zipformer-30M-RNNT-6000h, chạy
qua sherpa-onnx) — khối đầu tiên của Giai đoạn 1 (xem docs/roadmap.md mục 3).

Phát hiện quan trọng khi build: bản model công khai trên GitHub release của k2-fsa
(`sherpa-onnx-zipformer-vi-*`) là zipformer2 **non-streaming** (offline) — kiểm tra
bằng onnx.load(...).metadata_props thấy `comment = non-streaming zipformer2`, KHÔNG
phải bản streaming thật như mô tả trong roadmap. Dùng `sherpa_onnx.OnlineRecognizer`
(API streaming) sẽ crash native với lỗi thiếu metadata `encoder_dims`; phải dùng
`sherpa_onnx.OfflineRecognizer` (API batch) như dưới đây. Hệ quả: đây là
SegmentedSTTService (transcribe trọn câu khi VAD báo hết lượt nói), không phải true
streaming với partial transcript giữa câu. Bù lại decode rất nhanh trên CPU — đo được
~30x real-time (3.74s audio decode trong 0.09-0.11s) trên máy dev không GPU rời cho
việc này, nên không phải điểm nghẽn latency.

Tải model: `python scripts/download_asr_model.py` (một lần, ~33MB).
"""

import io
import wave
from typing import AsyncGenerator, Optional

import numpy as np
import sherpa_onnx
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class ZipformerVietnameseSTTService(SegmentedSTTService):
    """STT tiếng Việt tự host, offline-decode mỗi khi VAD báo người dùng ngừng nói."""

    def __init__(
        self,
        *,
        model_dir: str,
        num_threads: int = 2,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()

            with wave.open(io.BytesIO(audio), "rb") as wf:
                wav_sample_rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # accept_waveform tự resample nếu wav_sample_rate khác sample rate model
            # (16000Hz) nên không cần khớp tuyệt đối với audio_in_sample_rate của pipeline.
            stream = self._recognizer.create_stream()
            stream.accept_waveform(wav_sample_rate, samples)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()

            await self.stop_processing_metrics()

            if text:
                logger.debug(f"Zipformer transcription: [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
        except Exception as e:
            yield ErrorFrame(error=f"Zipformer STT error: {e}")
