# Speech2Speech (tiếng Việt)

Trợ lý giọng nói speech-to-speech tiếng Việt, xây theo lộ trình 4 giai đoạn trong
[`speech2speech.pdf`](../speech2speech.pdf) (bản đầy đủ, dễ đọc hơn ở
[`docs/roadmap.md`](docs/roadmap.md)).

## Đang ở giai đoạn nào

**Giai đoạn 0 — Khung xương (hoàn thành, chạy được đầy đủ).** Pipeline cascaded bằng
API thương mại: `WebRTC mic → Deepgram STT → Claude → ElevenLabs TTS → WebRTC loa`.
Mục tiêu: hệ thống streaming chạy được để chốt UX (barge-in, turn-taking), đo latency
từng chặng, làm eval harness trước khi thay từng khối bằng model tự host.

**Giai đoạn 1 — Tự host tiếng Việt (đang làm, một phần đã verify).** Chuyển từng khối
qua `STT_BACKEND`/`LLM_BACKEND`/`TTS_BACKEND=local` trong `.env`:

| Khối | Trạng thái |
|---|---|
| STT (Zipformer-30M-RNNT, sherpa-onnx) | ✅ Đã verify chạy tốt trên CPU |
| LLM (Qwen3 qua vLLM) | Code đã viết, chưa chạy được trên máy dev |
| TTS (F5-TTS-Vietnamese) | Code đã viết, chưa chạy được trên máy dev |

> Máy dev hiện tại (GPU 2GB VRAM) không đủ cho LLM/TTS tự host của giai đoạn 1 (cần
> ít nhất 1×RTX 4090/A10 + 1×A100/H100 theo lộ trình, mục 10 — Phần cứng), và vLLM
> không có wheel Windows (cần WSL2/Linux). Riêng ASR (model nhỏ, decode nhanh trên CPU)
> chạy tốt ngay trên máy này. Chi tiết đầy đủ, kể cả lý do kỹ thuật của từng giới hạn:
> [`docs/platform-architecture.md`](docs/platform-architecture.md) mục "Giai đoạn 1".

## Kiến trúc nền tảng

Ngoài pipeline giọng nói ở trên, repo còn có khung hạ tầng theo sơ đồ: Client SDK →
Gateway (Auth, Session Manager, Stream Multiplexer, Traffic Scheduler) → Event Bus
(NATS) → Runtime Scheduler (Ray) → 9 service (Speech, Vision, Text, Tool, Memory,
Planning, Reasoning, Generation, Monitoring). 7/9 service (trừ Speech, Monitoring) dùng
chung `services/llm_client.py` — provider-agnostic, đổi qua
`SERVICES_LLM_BACKEND=cloud|local` độc lập với backend của Speech Service. Chi tiết map
từng ô sang code, các quyết định kỹ thuật (vì sao NATS/Ray/audio không qua Event Bus),
và rủi ro đã biết nằm ở [`docs/platform-architecture.md`](docs/platform-architecture.md).

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# rồi điền ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID vào .env
```

Event Bus (NATS) chạy qua Docker:

```powershell
docker run -d --name nats -p 4222:4222 -p 8222:8222 nats:latest -js -m 8222
```

Muốn thử ASR tự host (Giai đoạn 1, đã verify chạy tốt trên CPU) thay vì Deepgram: tải
model rồi bật switch trong `.env`:

```powershell
.venv\Scripts\python scripts\download_asr_model.py
# rồi đặt STT_BACKEND=local trong .env (ASR_MODEL_DIR đã có sẵn giá trị mặc định khớp)
```

LLM (Qwen3/vLLM) và TTS (F5-TTS-Vietnamese) của Giai đoạn 1 đã có code
(`bot.py::build_llm`/`build_tts`) nhưng chưa chạy được trên máy dev hiện tại — xem
[`docs/platform-architecture.md`](docs/platform-architecture.md) mục "Giai đoạn 1" để
biết vì sao và cần gì để chạy thật (GPU đủ VRAM, WSL2/Linux cho vLLM).

## Chạy

Mở 3 terminal (đều `.venv\Scripts\python -m ...` để dùng đúng venv):

```powershell
python -m runtime.dispatcher            # Ray + 7 service actor (Vision/Text/Tool/Memory/Planning/Reasoning/Generation)
python -m services.monitoring_service   # log mọi event đi qua Event Bus
python -m gateway.main                  # Gateway: voice (WebRTC) + text/vision (WebSocket)
```

- Voice: mở `http://localhost:7860/client`, cấp quyền micro, và nói chuyện.
- Text/Vision: nối WebSocket tới `ws://localhost:7860/v1/ws` (mặc định không cần auth ở
  local dev — xem `GATEWAY_AUTH_DISABLED` trong `.env.example`), gửi
  `{"modality": "text", "content": "Xin chào"}`.

Vì sao Gateway tự dựng route WebRTC thay vì dùng thẳng `python bot.py -t webrtc` (cách
chuẩn của Pipecat)? Pipecat-ai bản mới nhất còn hỗ trợ Python 3.10 (0.0.108) có một bug
tương thích: `pipecat.runner.run` import `http.HTTPMethod`, chỉ có từ Python 3.11.
`gateway/main.py` dựng lại đúng phần cần thiết (route WebRTC offer/answer + giao diện
thử, cộng thêm auth/session/multiplexer) mà không đụng module đó. Nếu sau này nâng lên
Python 3.11+, có thể quay lại dùng `python bot.py -t webrtc` thẳng.

## Đo lường trước khi đổi model (`eval/`)

Console khi chạy voice sẽ in `[EVAL] TTFA`/`[EVAL] Turn latency` (mục tiêu < 1s theo lộ
trình) và transcript/TTFB từng service qua `MetricsLogObserver` — như trước. Giờ có
thêm 3 script đo có cấu trúc, chạy độc lập không cần mở trình duyệt:

```powershell
.venv\Scripts\python -m eval.asr_wer              # WER của STT_BACKEND đang cấu hình
.venv\Scripts\python -m eval.tool_call_accuracy   # LLM có gọi đúng tool không
.venv\Scripts\python -m eval.latency_report       # tổng hợp p50/p95 từ log hội thoại voice thật
```

`eval.asr_wer` đã chạy thật: **WER=0.000, ~124ms/câu** trên backend local (Zipformer) —
nhưng đọc kỹ [`docs/platform-architecture.md`](docs/platform-architecture.md) mục
"Track A" trước khi tin con số này, vì đây là TTS round-trip eval (audio tổng hợp bằng
gTTS, sạch hơn giọng người thật nhiều), không phải benchmark chuẩn. Đây là điều kiện để
trả lời câu hỏi "đổi model nhỏ hơn có giảm chất lượng nhiều không" bằng số liệu thay vì
đoán.

## Lưu ý chất lượng ở giai đoạn này

- ASR tiếng Việt qua Deepgram sẽ không tốt bằng Zipformer tự host của giai đoạn 1 —
  chấp nhận được, vì mục tiêu ở đây là đo UX/latency, không phải WER.
- ElevenLabs chỉ có `eleven_flash_v2_5` hỗ trợ tiếng Việt (không phải
  `eleven_multilingual_v2`) — đã set cứng trong `bot.py`.
- Barge-in/interruption dùng cơ chế mặc định của Pipecat (Silero VAD). Barge-in "đúng
  chuẩn" theo lộ trình (phân loại backchannel "dạ/ừm/vâng", AEC tường minh, phục hồi
  context khi bị ngắt) là việc của giai đoạn 1.

## Bước tiếp theo

LLM và TTS của Giai đoạn 1 cần một máy có GPU đủ VRAM (khuyến nghị theo roadmap:
1×RTX 4090/A10 + 1×A100/H100) và Linux/WSL2 cho vLLM để chạy thật — mã nguồn đã sẵn
sàng (`bot.py::build_llm`/`build_tts`, `selfhost/tts_server.py`), chỉ cần trỏ
`VLLM_BASE_URL`/`F5_TTS_*` trong `.env` khi có máy phù hợp. Sau đó là Giai đoạn 2 (full
duplex + cảm xúc) và Giai đoạn 3 (speech token, đích "như Qwen-Omni") — xem
[`docs/roadmap.md`](docs/roadmap.md) mục 2.
