# Speech2Speech (tiếng Việt)

Trợ lý giọng nói speech-to-speech tiếng Việt, xây theo lộ trình 4 giai đoạn trong
[`speech2speech.pdf`](../speech2speech.pdf) (bản đầy đủ, dễ đọc hơn ở
[`docs/roadmap.md`](docs/roadmap.md)).

## Đang ở giai đoạn nào

**Giai đoạn 0 — Khung xương (hoàn thành, chạy được đầy đủ).** Pipeline cascaded bằng
API thương mại: `WebRTC mic → Deepgram STT → Claude → ElevenLabs TTS → WebRTC loa`.
Mục tiêu: hệ thống streaming chạy được để chốt UX (barge-in, turn-taking), đo latency
từng chặng, làm eval harness trước khi thay từng khối bằng model tự host.

**Giai đoạn 1 — Tự host tiếng Việt (đã chạy end-to-end; đang tối ưu latency).** Chuyển từng khối
qua `STT_BACKEND`/`LLM_BACKEND`/`TTS_BACKEND=local` trong `.env`:

| Khối                                 | Trạng thái                                        |
| ------------------------------------- | --------------------------------------------------- |
| STT (Zipformer-30M-RNNT, sherpa-onnx) | ✅ Đã verify chạy tốt trên CPU                 |
| LLM (Qwen3-8B-AWQ qua vLLM)            | ✅ Đã verify chạy thật trên GPU, tool-calling 100% (5/5) — xem `docs/platform-architecture.md` |
| TTS (VieNeu-TTS true streaming)      | ✅ GPU sau warm-up: p50 first audio 207ms, p50 RTF 0.57; CPU không đạt ổn định — xem `selfhost/tts_server.py` |

> Máy dev tham chiếu ban đầu (GPU 2GB VRAM) không đủ cho LLM/TTS tự host — trên máy có
> GPU CUDA đủ VRAM, cả 3 khối local đã verify chạy thật. LLM qua vLLM cần venv riêng
> (`.venv-vllm`) và ~15 phút khởi động lần đầu (biên dịch kernel CUDA cho GPU đời mới —
> không phải lỗi, chỉ là chờ). Chi tiết đầy đủ, kể cả lý do kỹ thuật của từng giới hạn:
> [`docs/platform-architecture.md`](docs/platform-architecture.md) mục "Giai đoạn 1".

**Giai đoạn 2 — Full duplex + cảm xúc (mới viết, CHƯA verify bằng hội thoại thật).**
Code nằm ở [`duplex/`](duplex/), bật/tắt qua `.env` (mặc định bật phần đầu, tắt phần cảm
xúc):

| Mảnh                                                                      | Trạng thái                                                                                                                                                                     |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semantic endpointing (biết khi nào user nói XONG)                       | ✅ Hoá ra pipecat 0.0.108 đã dùng`LocalSmartTurnAnalyzerV3` (bundled ONNX, chạy CPU) làm mặc định — xác nhận bằng cách đọc source, không cần viết thêm gì |
| Phân loại backchannel + VAD barge-in khi bot đang nói | Unit test + benchmark mô phỏng 17/17; local ASR vẫn cắt bot sau 200ms dù không có interim transcript — chưa nghe thử qua mic thật |
| Phục hồi context khi bị ngắt lời giữa chừng                         | Code đã viết, verify plumbing tương tự — chưa quan sát thứ tự message thật khi có API key thật                                                                     |
| Kênh cảm xúc (`EMOTION_BACKEND=heuristic`)                            | Chỉ là khớp từ khoá trên transcript, KHÔNG phải SER thật (roadmap.md mục 6 đề xuất emotion2vec+/SenseVoice trên audio — chưa tích hợp)                         |

> "Verify plumbing" ở đây nghĩa là đã `import bot` và dựng thật `Pipeline([...])` với
> `ANTHROPIC_API_KEY`/`ELEVENLABS_API_KEY` giả trong `.venv` có cài `pipecat-ai` thật —
> xác nhận toàn bộ object graph khớp API thật của thư viện, chưa chạy một cuộc hội thoại
> voice thật để nghe kết quả. Xem docstring từng file trong `duplex/` để biết chi tiết.

**Giai đoạn 3 — chuẩn bị dữ liệu (chưa đụng tới phần fine-tune model).**
[`datagen/synthetic_dialogue.py`](datagen/synthetic_dialogue.py) sinh hội thoại full-
duplex 2 kênh tiếng Việt tổng hợp (LLM viết kịch bản có ngắt lời/backchannel → TTS 2
giọng → ghép stereo có chồng lấn thời gian thật) — đúng đề xuất ở
[`docs/roadmap.md`](docs/roadmap.md) mục 8. Không cần GPU, không phụ thuộc việc chọn
hướng (a)/(b)/(c) nào ở mục 2 của roadmap. Logic ghép audio đã có unit test bằng audio
giả; phần gọi API thật (Claude sinh kịch bản + ElevenLabs tổng hợp giọng) chưa chạy vì
tốn phí mỗi lần — chạy thử một mẻ nhỏ (`--n 1`) trước khi sinh số lượng lớn.

Đã chọn hướng (a) — fine-tune Talker Qwen3-Omni nói tiếng Việt.
[`datagen/talker_finetune_corpus.py`](datagen/talker_finetune_corpus.py) chuẩn bị
corpus audio-text cho hướng này (vivos/viVoice/PhoAudiobook → 16kHz + manifest
JSONL). ĐÃ CHẠY THẬT toàn bộ vivos (công khai, ~15h) vào `data/talker_corpus/vivos/`.
viVoice và PhoAudiobook — hai nguồn chính, lớn hơn nhiều — đều **bị gate trên
HuggingFace** (yêu cầu tự xin quyền bằng email trường/công ty), chưa chạy được vì cần
người dùng tự duyệt. Quan trọng hơn: **máy hiện tại (1×GB10 dùng chung) không đủ quy
mô cho bước train thật** — roadmap tự ước tính cần ~8×A100 80GB, xem
`docs/platform-architecture.md` mục Track D để biết số đo thật. Bước train sẽ cần
quyết định thuê GPU cloud riêng, chưa làm ở bước này.

## Kiến trúc nền tảng

Ngoài pipeline giọng nói ở trên, repo còn có khung hạ tầng theo sơ đồ: Client SDK →
Gateway (Auth, Session Manager, Stream Multiplexer, Traffic Scheduler) → Event Bus
(NATS) → Runtime Scheduler (dispatcher asyncio local hoặc Ray khi scale nhiều process/máy)
→ 9 service (Speech, Vision, Text, Tool, Memory,
Planning, Reasoning, Generation, Monitoring). 7/9 service (trừ Speech, Monitoring) dùng
chung `services/llm_client.py` — provider-agnostic, đổi qua
`SERVICES_LLM_BACKEND=cloud|local` độc lập với backend của Speech Service. Chi tiết map
từng ô sang code, các quyết định kỹ thuật (vì sao audio không qua Event Bus),
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

Muốn thử TTS tự host (VieNeu-TTS, đã verify chạy thật cả CPU lẫn GPU) thay vì
ElevenLabs: cài venv riêng rồi bật switch — model tự tải về ở lần chạy server đầu tiên,
không cần script tải riêng:

```bash
python -m venv .venv-tts && .venv-tts/bin/pip install -r requirements-tts.txt
.venv-tts/bin/python selfhost/tts_server.py   # chạy server TTS ở terminal riêng, cổng 8100
# rồi đặt TTS_BACKEND=local trong .env (các biến VIENEU_* đã có sẵn giá trị mặc định khớp)
# Trên máy có đủ GPU: VIENEU_DEVICE=cuda để phát liền mạch; luôn bật VIENEU_STREAMING=true
.venv/bin/python -m eval.tts_streaming_latency --assert-realtime
```

Muốn thử LLM tự host (Qwen3-8B-AWQ qua vLLM, đã verify chạy thật trên GPU, tool-calling
100%) thay vì Claude: cài venv riêng rồi chạy server (lần đầu mất ~15 phút biên dịch
kernel CUDA — không phải lỗi, chỉ là chờ):

```bash
python -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements-vllm.txt
.venv-vllm/bin/vllm serve Qwen/Qwen3-8B-AWQ \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --gpu-memory-utilization 0.15 --max-model-len 8192
# rồi đặt LLM_BACKEND=local trong .env (VLLM_MODEL đã có sẵn giá trị mặc định khớp)
```

`--gpu-memory-utilization` phải nhỏ hơn tỉ lệ bộ nhớ GPU thực sự còn trống lúc chạy
(kiểm tra bằng `nvidia-smi`/`free -h` trước). `--default-chat-template-kwargs` **bắt
buộc** — thiếu cờ này Qwen3 tự sinh rất nhiều token suy luận `<think>` trước khi trả
lời (đã đo thật: ~229 token/lượt, nhiều giây), quá chậm cho voice chat; có cờ thì chỉ
~60-70 token/lượt và tool-calling vẫn đúng 100%. Nếu `TTS_BACKEND=local`, chỉ đặt
`VIENEU_DEVICE=cuda` khi còn đủ bộ nhớ sau khi vLLM load; trên máy GB10 hiện tại đã đo
p50 first audio 207ms và RTF 0.57. CPU là fallback an toàn nhưng lần đo mới chỉ đạt p50
first audio khoảng 535ms, RTF 1.79 nên câu dài có thể bị hụt buffer. Xem
[`docs/platform-architecture.md`](docs/platform-architecture.md) mục "Giai đoạn 1" để
biết chi tiết đầy đủ.

## Chạy

### Local small/medium — cách khuyến nghị

Launcher kiểm tra dependency, tự dùng lại NATS/vLLM/VieNeu đang khỏe và chỉ dừng
những process do chính nó tạo. Profile local dùng dispatcher asyncio một process thay
cho Ray, phù hợp máy đơn và không cấp trước object store lớn:

```bash
.venv/bin/python -m scripts.local_stack doctor --profile small
.venv/bin/python -m scripts.local_stack start --profile small
.venv/bin/python -m scripts.local_stack status
.venv/bin/python -m scripts.local_smoke          # gọi thật Qwen + VieNeu
```

Sau khi smoke test đạt, mở `http://localhost:7860/client/`, cấp quyền micro và bấm
Connect. Dashboard/Settings ở `http://localhost:5173`. Dừng các process do launcher
quản lý bằng:

```bash
.venv/bin/python -m scripts.local_stack stop
```

Đổi `small` thành `medium` khi cần nhiều phiên đồng thời hơn. Hướng dẫn đầy đủ về
profile, log, kiểm tra microphone, readiness và lộ trình nâng từ `shadow` lên audio-token
nằm ở [`docs/local-deployment.md`](docs/local-deployment.md).

### Cách chạy cũ / phát triển hạ tầng Ray

Script cũ vẫn còn cho trường hợp cần Ray, monitoring và muốn chạy toàn bộ stack theo
cấu hình hạ tầng:

```bash
./scripts/run_all.sh
```

Hoặc mở các terminal riêng:

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

Gateway giữ route WebRTC riêng để gắn auth/session/multiplexer. Route này hỗ trợ đúng
hai bước mà UI Pipecat dựng sẵn gọi: `POST /start`, rồi
`/sessions/{session_id}/api/offer`; xem `gateway/main.py`.

## Đo lường trước khi đổi model (`eval/`)

Console khi chạy voice sẽ in `[EVAL] TTFA`/`[EVAL] Turn latency` (mục tiêu < 1s theo lộ
trình) và transcript/TTFB từng service qua `MetricsLogObserver` — như trước. Giờ có
thêm các script đo có cấu trúc, chạy độc lập không cần mở trình duyệt:

```powershell
.venv\Scripts\python -m eval.asr_wer              # WER của STT_BACKEND đang cấu hình
.venv\Scripts\python -m eval.tool_call_accuracy   # LLM có gọi đúng tool không
.venv\Scripts\python -m eval.latency_report       # tổng hợp p50/p95 từ log hội thoại voice thật
.venv\Scripts\python -m eval.tts_streaming_latency --assert-realtime
.venv\Scripts\python -m eval.run_benchmarks       # turn-taking + latency hiện có
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
- Barge-in dùng Silero VAD với cửa sổ phân loại backchannel 200ms. Đây là full-duplex ở
  mức turn-taking/cắt lời; chưa phải speech-native full duplex kiểu Moshi.
- Zipformer local vẫn là ASR theo đoạn: chỉ trả transcript sau khi VAD kết thúc lượt nói.
  Vì vậy hệ thống hiện là cascaded real-time, chưa phải mọi khối đều streaming từng frame.

## Bước tiếp theo

Cả 3 khối của Giai đoạn 1 (STT/LLM/TTS) đã verify chạy thật. Giai đoạn 2/3 cũ (full
duplex thuần + fine-tune Talker) đã được thay bằng kiến trúc **Dual-path Anchored
Speech-to-Speech** (pivot ghi ngày 2026-07-17) — xem
[`docs/roadmap.md`](docs/roadmap.md) mục 11. Cascaded pipeline hiện tại không bị bỏ,
trở thành Anchor path + Cascade fallback trong kiến trúc mới.

Nền control/media plane của pipeline mới đã được triển khai trong `s2s/`:

- `AudioHub`: ring buffer và fan-out audio trong process; consumer chậm tự drop frame
  của mình, không back-pressure WebRTC/ASR.
- `SessionOrchestrator`: nguồn state duy nhất cho `turn_id`, `revision_id`,
  `generation_id`, cancellation và playback state.
- `SegmentPolicy`: hard gate deterministic cho stale revision, entity/factual prefix và
  tool dependency.
- `AudioTokenS2SBackend`: interface model-neutral; có `probe` và adapter `moshi_ws`
  theo đúng WebSocket/Opus 24 kHz của Kyutai. Moshi vẫn chạy ở process/venv riêng.
- `S2S_MODE=off|shadow|ack_only|speculative|primary`; mặc định `shadow`, không thay đổi
  audio/câu trả lời production.

Pipeline runtime hiện tại:

```text
WebRTC -> AudioHub -> Zipformer/Deepgram -> Qwen3/Claude -> VieNeu/ElevenLabs -> WebRTC
                \\-> probe/Moshi adapter (shadow) -> SegmentPolicy -----------/
```

Mimi codec thật đã được smoke-test trên ba speaker VIVOS người thật: cả ba có WER delta
bằng 0; GPU sau warm-up có RTF 0,18-0,46 (đạt realtime), trong khi CPU mẫu đầu có RTF
7,97. Đây mới là smoke test ba mẫu, chưa thay thế benchmark 300-500 câu/giọng vùng miền.
Moshi checkpoint đầy đủ chưa
được bật vì GPU đang dùng chung; adapter/mock protocol đã test, mặc định vẫn `probe`.
Checklist
chi tiết nằm trong
[`docs/ke-hoach-cong-viec-train-test-s2s.md`](docs/ke-hoach-cong-viec-train-test-s2s.md).

Mimi/Moshi dùng venv riêng; không cài PyTorch vào Gateway venv:

```bash
./scripts/setup_s2s_env.sh
.venv-s2s/bin/python -m eval.mimi_codec --input <audio-vi.wav> \
  --output eval/results/mimi-reconstructed.wav \
  --result-json eval/results/mimi-codec.json --device cuda
.venv/bin/python -m eval.mimi_asr_compare \
  --original <audio-vi.wav> --reconstructed eval/results/mimi-reconstructed.wav \
  --transcript '<transcript chuẩn>'
```

Khi có ít nhất khoảng 24 GB GPU trống riêng cho Moshi, có thể dùng
`scripts.local_stack start --profile small --with-moshi`. Cờ này mới tải/chạy checkpoint
và ép `shadow`; start bình thường không tải model hoặc chiếm GPU. Checkpoint gốc chủ yếu
tiếng Anh, tuyệt đối chưa chuyển sang `ack_only/speculative/primary` cho tiếng Việt.
