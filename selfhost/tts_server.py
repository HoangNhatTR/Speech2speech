"""Server nội bộ cho F5-TTS-Vietnamese-ViVoice (hynt/F5-TTS-Vietnamese-ViVoice) — khối
TTS tự host của Giai đoạn 1 (docs/roadmap.md mục 3 và 7).

QUAN TRỌNG — venv riêng: file này KHÔNG chạy trong .venv của dự án (nơi chạy
gateway/bot.py). Dependency của f5-tts (torch, transformers, gradio, hydra-core...) rất
nặng và có nguy cơ xung đột phiên bản fastapi/starlette với Pipecat Gateway nếu cài
chung venv (repo này từng dính đúng kiểu xung đột này với global Python env — xem lịch
sử commit). Nên f5-tts sống trong .venv-tts riêng, giao tiếp với bot.py qua HTTP nội bộ
(loopback, không qua Event Bus vì đây là 1 request/response đơn giản, không phải audio
streaming thời gian thực).

Cài đặt (chạy 1 lần):
    python -m venv .venv-tts
    .venv-tts\\Scripts\\pip install f5-tts fastapi uvicorn python-multipart

Chạy: .venv-tts\\Scripts\\python selfhost/tts_server.py

F5-TTS là voice cloning zero-shot — BẮT BUỘC có audio + text tham chiếu (không có
giọng mặc định dựng sẵn). Đặt F5_TTS_REF_AUDIO/F5_TTS_REF_TEXT trỏ tới giọng bạn có
quyền sử dụng (đã xin phép người nói) — đúng yêu cầu "consent flow khi enroll giọng"
trong docs/roadmap.md mục 7. KHÔNG dùng giọng người khác khi chưa có sự đồng ý.

Đã KIỂM TRA nhưng CHƯA CHẠY THỬ được: checkpoint model_last.pt nặng ~5GB, máy dev chỉ
có CPU (torch cài mặc định là bản CPU-only) và GPU 2GB không đủ VRAM — tải + suy luận
CPU với model ~300M tham số kiểu flow-matching (nhiều bước khử nhiễu) sẽ rất chậm, có
thể mất hàng chục giây tới vài phút mỗi câu. Code này viết đúng theo API thật của
f5_tts.api.F5TTS (đã đọc source), nhưng cần máy có GPU đủ VRAM để dùng được trong thực
tế — xem docs/platform-architecture.md.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

_tts = None


class SynthesizeRequest(BaseModel):
    text: str


def _load_model():
    global _tts
    if _tts is not None:
        return _tts

    from f5_tts.api import F5TTS

    ckpt_file = os.environ["F5_TTS_CKPT_FILE"]
    vocab_file = os.environ["F5_TTS_VOCAB_FILE"]
    _tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_file, vocab_file=vocab_file)
    return _tts


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> Response:
    ref_audio = os.getenv("F5_TTS_REF_AUDIO")
    ref_text = os.getenv("F5_TTS_REF_TEXT")
    if not ref_audio or not ref_text:
        raise HTTPException(
            status_code=500,
            detail=(
                "Thiếu F5_TTS_REF_AUDIO/F5_TTS_REF_TEXT — cần audio+text tham chiếu "
                "của một giọng bạn có quyền sử dụng (xem docstring đầu file)."
            ),
        )

    import io

    import soundfile as sf

    tts = _load_model()
    wav, sample_rate, _ = tts.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=request.text)

    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print("Đang tải model F5-TTS-Vietnamese (lần đầu có thể mất vài phút)...")
    _load_model()
    print("Model sẵn sàng. Server chạy tại http://localhost:8100")
    uvicorn.run(app, host="localhost", port=8100)
