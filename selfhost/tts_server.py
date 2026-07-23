"""Server nội bộ cho VieNeu-TTS-v3-Turbo (pnnbao-ump/VieNeu-TTS-v3-Turbo trên HuggingFace,
gói pip `vieneu`) — khối TTS tự host của Giai đoạn 1 (docs/roadmap.md mục 3 và 7). Thay
thế F5-TTS-Vietnamese-ViVoice trước đây (xem git history nếu cần code cũ) — VieNeu-TTS
nhẹ hơn nhiều để cài (không bắt buộc torch cho đường CPU, dùng ONNX Runtime int8), có 14
giọng dựng sẵn (không bắt buộc audio tham chiếu như F5-TTS), có watermark âm thanh tích
hợp (Perth), và giấy phép Apache-2.0 (thoáng hơn cc-by-nc-sa-4.0 của F5-TTS-Vietnamese-
ViVoice). Chọn cụ thể bản v3-Turbo (48kHz, nhanh nhất trong 3 bản v1/v2/v3-Turbo của
pnnbao97) — xem VIENEU_BACKBONE_REPO nếu muốn thử bản khác.

QUAN TRỌNG — venv riêng: file này KHÔNG chạy trong .venv của dự án (nơi chạy
gateway/bot.py), lý do tương tự F5-TTS trước đây: cô lập dependency nặng/không chắc
tương thích khỏi Pipecat Gateway (đã dính xung đột `fastapi`/`starlette` với global
Python env trước đây — xem lịch sử commit). Giao tiếp với bot.py qua HTTP nội bộ
(loopback, không qua Event Bus vì đây là 1 request/response đơn giản).

Cài đặt (chạy 1 lần):
    python -m venv .venv-tts
    .venv-tts/bin/pip install -r requirements-tts.txt   # Linux/macOS
    .venv-tts\\Scripts\\pip install -r requirements-tts.txt   # Windows

    # Đường CPU (mặc định, torch-free, ONNX int8) không cần cài gì thêm. Muốn dùng GPU
    # (PyTorch, nhanh hơn khi có nhiều câu dồn dập — xem benchmark trong docstring dưới):
    .venv-tts/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    # (trên máy đã có torch+CUDA cài sẵn từ trước, ví dụ cùng .venv-tts từng cài F5-TTS,
    # VieNeu-TTS tự dùng GPU đó, không cần cài lại — chỉ khi .venv-tts hoàn toàn mới mới
    # cần lệnh trên.)

Model tự tải về (qua huggingface_hub, cache tại ~/.cache/huggingface/) ở lần chạy đầu
tiên — KHÔNG cần script tải riêng như F5-TTS trước đây.

Chạy: .venv-tts/bin/python selfhost/tts_server.py (Linux/macOS) hoặc
      .venv-tts\\Scripts\\python selfhost/tts_server.py (Windows)

Giọng nói: mặc định dùng một trong 14 giọng dựng sẵn của VieNeu-TTS (đặt qua
VIENEU_VOICE trong .env, xem .env.example để xem danh sách tên giọng) — KHÔNG cần audio
tham chiếu tự cung cấp như F5-TTS. Muốn voice cloning giọng riêng: đặt VIENEU_REF_AUDIO
trỏ tới audio bạn có quyền sử dụng (đã xin phép người nói) — ưu tiên hơn VIENEU_VOICE
nếu cả hai cùng đặt.

Server cung cấp hai endpoint:
- `/synthesize/stream`: đường mặc định cho hội thoại real-time. Dùng
  `Vieneu.infer_stream()` và trả raw PCM16 mono 48 kHz ngay khi model sinh được chunk
  đầu tiên; không chờ toàn bộ câu hoàn tất.
- `/synthesize`: đường tương thích cũ, trả một WAV hoàn chỉnh sau khi tổng hợp cả câu.

ĐÃ VERIFY chạy thật, cả hai backend:
- CPU (ONNX int8): lần đo mới qua HTTP streaming p50 first audio ~535ms, p50 RTF ~1.79.
- GPU (PyTorch): sau warm-up p50 first audio 207ms, p50 RTF 0.57 (3 câu đo).

QUAN TRỌNG — đã xảy ra thật: nếu LLM_BACKEND=local (vLLM) cùng chạy trên máy này,
VIENEU_DEVICE=auto khiến TTS cũng giành GPU với vLLM và **crash "CUDA error: out of
memory"** ngay khi có phiên hội thoại thật (vLLM chiếm phần lớn VRAM trước, TTS load
model sau nên hết bộ nhớ). Mặc định ở đây đã đổi thành "cpu" — chỉ đặt "auto"/"cuda" nếu
chắc chắn máy có đủ VRAM cho CẢ HAI cùng lúc.
"""

import io
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

load_dotenv(override=True)

_tts = None
_model_lock = threading.Lock()
_warmup_lock = threading.Lock()
_warmed_up = False
logger = logging.getLogger("speech2speech.tts")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warm_up_model()
    yield


app = FastAPI(lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str


def _load_model():
    global _tts
    if _tts is not None:
        return _tts

    # Tránh hai request đầu tiên cùng tải hai bản model vào RAM/VRAM.
    with _model_lock:
        if _tts is not None:
            return _tts

        from vieneu import Vieneu

        # Ghi tường minh mode/backbone_repo thay vì để mặc định ngầm của thư viện
        # `vieneu` — tránh hành vi đổi lặng lẽ nếu bản pip sau này đổi default.
        _tts = Vieneu(
            mode="v3turbo",
            backbone_repo=os.getenv(
                "VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo"
            ),
            device=os.getenv("VIENEU_DEVICE", "cpu"),
        )
    return _tts


def _float_audio_to_pcm16(wav: np.ndarray) -> bytes:
    """Đổi waveform float [-1, 1] sang PCM16 little-endian dùng trên loopback HTTP."""
    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _stream_pcm16(tts, request: SynthesizeRequest) -> Iterator[bytes]:
    """Generator đồng bộ; Starlette chạy nó trong thread pool và flush từng chunk."""
    ref_audio = os.getenv("VIENEU_REF_AUDIO")
    voice = os.getenv("VIENEU_VOICE")
    started_at = time.perf_counter()
    first_chunk = True

    kwargs = {"ref_audio": ref_audio} if ref_audio else {"voice": voice or None}
    for wav_chunk in tts.infer_stream(request.text, **kwargs):
        pcm = _float_audio_to_pcm16(wav_chunk)
        if not pcm:
            continue
        if first_chunk:
            first_chunk = False
            logger.info(
                "VieNeu-TTS first audio: %.0f ms (%d bytes)",
                (time.perf_counter() - started_at) * 1000,
                len(pcm),
            )
        yield pcm


def _warm_up_model() -> None:
    """Chạy một câu ngắn trước khi nhận traffic để lượt hội thoại đầu không trả giá compile."""
    global _warmed_up
    if _warmed_up or os.getenv("VIENEU_WARMUP", "true").strip().lower() != "true":
        return

    with _warmup_lock:
        if _warmed_up:
            return
        tts = _load_model()
        voice = os.getenv("VIENEU_VOICE")
        started_at = time.perf_counter()
        try:
            for _ in tts.infer_stream("Xin chào.", voice=voice or None):
                pass
            _warmed_up = True
            logger.info(
                "VieNeu-TTS warm-up hoàn tất trong %.1fs",
                time.perf_counter() - started_at,
            )
        except Exception:
            # Warm-up là tối ưu latency, không phải điều kiện để server sống. Request thật
            # vẫn trả lỗi chi tiết nếu model thực sự không inference được.
            logger.exception("VieNeu-TTS warm-up lỗi; server vẫn tiếp tục khởi động")


@app.get("/health")
def health() -> dict:
    tts = _load_model()
    return {
        "status": "ready",
        "sample_rate": tts.sample_rate,
        "streaming": callable(getattr(tts, "infer_stream", None)),
        "warmed_up": _warmed_up,
    }


@app.post("/synthesize/stream")
def synthesize_stream(request: SynthesizeRequest) -> StreamingResponse:
    """True streaming: raw PCM16 được flush ngay khi VieNeu sinh ra từng audio chunk."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text không được để trống")
    tts = _load_model()
    return StreamingResponse(
        _stream_pcm16(tts, request),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Sample-Rate": str(tts.sample_rate),
            "X-Audio-Channels": "1",
            "Cache-Control": "no-store",
        },
    )


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest) -> Response:
    """Endpoint WAV cũ để tương thích/rollback; chạy sync trong FastAPI thread pool."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text không được để trống")
    tts = _load_model()

    ref_audio = os.getenv("VIENEU_REF_AUDIO")
    voice = os.getenv("VIENEU_VOICE")

    try:
        if ref_audio:
            wav = tts.infer(request.text, ref_audio=ref_audio)
        else:
            wav = tts.infer(request.text, voice=voice or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VieNeu-TTS lỗi tổng hợp: {e}") from e

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav, tts.sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print("Đang tải và warm-up VieNeu-TTS (lần đầu có thể mất vài phút)...")
    _warm_up_model()
    print("Model sẵn sàng. Server chạy tại http://localhost:8100")
    uvicorn.run(app, host="localhost", port=8100)
