"""ASR tự host cho tiếng Việt — khối đầu tiên của Giai đoạn 1 (xem docs/roadmap.md mục
3). Hai lựa chọn engine, chọn qua ASR_LOCAL_ENGINE trong .env (xem bot.py::build_stt):

  zipformer (mặc định) — Zipformer train tiếng Việt trên ~70.000 giờ
    (zzasdf/viet_iter3_pseudo_label, qua k2-fsa/sherpa-onnx). ĐÃ ĐO: WER tổng 2.7%,
    latency ~65ms/câu. Trước đây dùng bản 30M train 6000 giờ (WER 3.5%, ~40ms/câu) —
    đổi sang bản này sau khi có phản hồi thật: bản cũ nhận dạng sai nhiều trên hội
    thoại thật qua mic dù benchmark tổng hợp (gTTS) không cho thấy rõ mức độ (benchmark
    quá "sạch" so với giọng nói tự nhiên/có nhiễu — xem eval/testset.py). Muốn quay lại
    bản nhẹ hơn: `python scripts/download_asr_model.py --variant small`.
  whisper — Whisper large-v3-turbo (ONNX qua sherpa-onnx), model đa ngôn ngữ tổng
    quát. ĐÃ ĐO: WER tổng 12.5% (thua Zipformer mọi domain trừ code-switch),
    latency ~2474ms/câu (~38x chậm hơn) — KHÔNG khuyến nghị làm mặc định, xem
    WhisperTurboVietnameseSTTService bên dưới để biết bảng so sánh đầy đủ.

Cả hai đều chạy qua sherpa-onnx (đã có sẵn trong .venv chính, không cần thêm runtime).

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

Tải model: `python scripts/download_asr_model.py` (một lần, ~68MB, mặc định bản 70k giờ).
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


class WhisperTurboVietnameseSTTService(SegmentedSTTService):
    """STT tự host bằng Whisper large-v3-turbo (ONNX, qua sherpa-onnx) — lựa chọn ASR
    local THỨ HAI bên cạnh Zipformer, xem docstring đầu file cho bối cảnh so sánh.

    Whisper large-v3-turbo là model đa ngôn ngữ tổng quát (809M tham số, decoder rút từ
    32 xuống 4 layer so với large-v3 gốc) — KHÔNG train riêng cho tiếng Việt như
    Zipformer (chỉ 30M tham số, train trên 6000 giờ tiếng Việt). OpenAI tự ghi nhận bản
    turbo giảm độ chính xác nhiều hơn ở ngôn ngữ ít tài nguyên so với bản gốc.

    ĐÃ ĐO THẬT bằng eval/asr_wer.py trên tập categorized (63 câu, 6 domain × 3 mức
    nhiễu) — kết luận: Zipformer tốt hơn rõ rệt cho tiếng Việt, KHÔNG khuyến nghị
    Whisper-turbo làm mặc định:

    | | Zipformer | Whisper-turbo |
    |---|---|---|
    | WER tổng | **3.5%** | 12.5% |
    | WER so_thoi_gian (số/giờ) | 0.0% | **42.2%** (điểm yếu nặng nhất) |
    | WER code_switch (VN-EN) | 18.1% | **15.2%** (điểm mạnh duy nhất) |
    | WER trung bình theo nhiễu (15dB/5dB) | 3.1-4.3% | 12.8-13.6% |
    | Latency trung bình/câu | 40ms | **2474ms** (~62x chậm hơn) |

    Whisper-turbo chỉ nhỉnh hơn ở code-switching (cả hai đều chưa tốt), còn lại thua
    toàn diện — đặc biệt latency 2.5s/câu không phù hợp voice chat thời gian thực. Giữ
    lại làm lựa chọn CÓ THỂ BẬT (không phải mặc định) cho trường hợp cần thử nghiệm/so
    sánh, không khuyến nghị dùng cho production tại thời điểm đo (xem
    eval/results/asr_wer_*.json để xem chi tiết từng câu).

    Cũng là non-streaming (offline decode trọn câu khi VAD báo hết lượt nói), giống
    Zipformer — không phải điểm khác biệt giữa hai lựa chọn.

    provider="cpu" cố định: máy test không có `onnxruntime-gpu` (chỉ
    CPUExecutionProvider/AzureExecutionProvider khả dụng) — muốn dùng GPU cần cài
    onnxruntime-gpu và xác nhận sherpa-onnx nhận đúng CUDAExecutionProvider trước khi đổi
    (có thể cải thiện latency, nhưng không thay đổi kết luận về WER ở trên).

    Tải model: `python scripts/download_whisper_asr_model.py` (một lần, ~1GB).
    """

    def __init__(
        self,
        *,
        model_dir: str,
        language: str = "vi",
        num_threads: int = 2,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=f"{model_dir}/turbo-encoder.int8.onnx",
            decoder=f"{model_dir}/turbo-decoder.int8.onnx",
            tokens=f"{model_dir}/turbo-tokens.txt",
            language=language,
            task="transcribe",
            num_threads=num_threads,
            provider="cpu",
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()

            with wave.open(io.BytesIO(audio), "rb") as wf:
                wav_sample_rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            stream = self._recognizer.create_stream()
            stream.accept_waveform(wav_sample_rate, samples)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()

            await self.stop_processing_metrics()

            if text:
                logger.debug(f"Whisper-turbo transcription: [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
        except Exception as e:
            yield ErrorFrame(error=f"Whisper-turbo STT error: {e}")
