# Kiến trúc nền tảng: Gateway / Event Bus / Runtime Scheduler / 9 service

Tài liệu này map từng ô trong sơ đồ kiến trúc gốc sang code/công nghệ thật trong repo,
giải thích các quyết định lệch so với đọc sơ đồ theo nghĩa đen, và đường mở rộng khi cần
scale vượt quá một máy dev.

## Bản đồ sơ đồ -> code

| Ô trong sơ đồ | Công nghệ | Code |
|---|---|---|
| Client SDK (WebRTC) | Pipecat SmallWebRTC + prebuilt UI | `bot.py`, mount trong `gateway/main.py` |
| Client SDK (WebSocket) | FastAPI WebSocket | `gateway/main.py` (`/v1/ws`) |
| Gateway → Authentication | API key tĩnh qua header/query | `gateway/auth.py` |
| Gateway → Session Manager | dict in-memory | `gateway/session_manager.py` |
| Gateway → Stream Multiplexer | điều hướng theo `modality` | `gateway/multiplexer.py` |
| Gateway → Traffic Scheduler | `asyncio.Semaphore` (per-session + toàn cục) | `gateway/multiplexer.py` |
| Event Bus | NATS (JetStream), chạy qua Docker | `eventbus/client.py` |
| Runtime Scheduler | Ray, local mode (`ray.init()`) | `runtime/dispatcher.py` |
| Speech Service | pipeline Pipecat (Giai đoạn 0 cũ) | `bot.py` |
| Vision / Text / Tool / Memory / Planning / Reasoning / Generation | Ray actor, bản mỏng gọi Claude | `services/*.py` |
| Monitoring Service | subscriber thụ động trên Event Bus | `services/monitoring_service.py` |
| GPU Cluster | **không có** — mọi cognitive service gọi Claude API | `services/anthropic_client.py` |

## Ba quyết định lệch khỏi đọc sơ đồ theo nghĩa đen

**1. NATS thay vì Kafka.** Sơ đồ gốc liệt kê Kafka/NATS/Redis Streams như 3 lựa chọn
tương đương. Máy dev hiện chỉ có ~2.1GB RAM trống; Kafka cần JVM (thường khuyến nghị
≥1-2GB heap) trong khi NATS chỉ tốn ~10-20MB. NATS cũng có request-reply built-in
(`nc.request()`), khớp thẳng với mô hình "Gateway gọi service, đợi kết quả". Đổi sang
Kafka sau này (khi cần throughput lớn, nhiều consumer group, replay dài hạn) chỉ cần
viết lại `eventbus/client.py` — toàn bộ `gateway/`, `runtime/`, `services/`, `bot.py`
không cần đổi vì chúng chỉ gọi `publish()`/`request()`/`subscribe()`.

**2. Ray chạy local mode, không phải cluster thật.** Không có GPU cluster thật vì
không service nào tự host model — tất cả gọi Claude API. `ray.init()` dùng
multiprocessing trên chính máy dev. Khi thật sự cần scale-out (nhiều máy, nhiều GPU tự
host ở giai đoạn 3 của `roadmap.md`), đổi sang `ray.init(address="ray://...")` trỏ tới
cluster thật — code trong `services/` không cần đổi vì chúng không biết gì về Ray, chỉ
`runtime/dispatcher.py` mới biết.

**3. Audio thời gian thực không đi qua Event Bus.** Đẩy từng frame audio (~20ms/lần)
qua NATS + Ray remote call sẽ cộng thêm latency đúng vào chỗ roadmap giọng nói đang cố
tối ưu (ngân sách P50 ~600-950ms, barge-in <150ms — xem `roadmap.md` mục 4). Vì vậy
Speech Service là ngoại lệ duy nhất: nó chạy trong-tiến-trình bên trong Gateway
(`bot.py`, khởi động qua `gateway/main.py`'s `/api/offer`), không phải một Ray actor
đứng sau `runtime/dispatcher.py`. Khi cần gọi service khác — ví dụ tool calling — Speech
Service publish request lên Event Bus giống hệt mọi client khác (xem
`handle_get_current_time` trong `bot.py`, gọi `svc.tool`). Đây chính là mẫu "LLM anchor"
trong roadmap: hệ thống ngoài can thiệp giữa bước sinh text và bước sinh speech mà không
phá vỡ ngân sách latency của đường audio.

## Vì sao 7/9 service ban đầu chỉ là Claude API

Vision/Text/Tool/Memory/Planning/Reasoning/Generation chưa có model chuyên biệt tự host
(việc đó là giai đoạn 3 của `roadmap.md`, cần GPU cloud thuê theo đợt). Mục tiêu của lần
build này là chứng minh **toàn bộ topology chạy được** — mọi ô trong sơ đồ có code thật,
nói chuyện được với nhau qua đúng lớp — trước khi đào sâu logic riêng từng service.
`services/anthropic_client.py` là điểm dùng chung; khác biệt giữa các service hiện tại
chỉ là system prompt.

## Cách chạy

```powershell
docker run -d --name nats -p 4222:4222 nats:latest -js
.venv\Scripts\python -m runtime.dispatcher       # Ray + 7 service actor
.venv\Scripts\python -m services.monitoring_service  # log mọi event
.venv\Scripts\python -m gateway.main              # Gateway (voice + /v1/ws)
```

Test nhánh text qua `/v1/ws` (vd bằng `websocat` hoặc script Python nhỏ):

```json
{"modality": "text", "content": "Xin chào"}
```

## Giai đoạn 1 — tự host tiếng Việt (thay từng khối trong Speech Service)

Bật qua `.env`: `STT_BACKEND` / `LLM_BACKEND` / `TTS_BACKEND` = `cloud` (mặc định) hoặc
`local`. Phần còn lại của pipeline (VAD, tool calling, eval harness) không đổi khi swap
— đúng tinh thần "mỗi khối thay được" của `roadmap.md`.

| Khối | Model | Trạng thái | Code |
|---|---|---|---|
| STT | Zipformer-30M-RNNT (hynt, qua sherpa-onnx) | **Đã verify chạy** trên CPU, ~30x real-time | `selfhost/asr.py` |
| LLM | Qwen3 qua vLLM (API tương thích OpenAI) | Viết code, **chưa chạy được** trên máy dev | cấu hình trong `bot.py::build_llm` |
| TTS | F5-TTS-Vietnamese-ViVoice (hynt, venv riêng) | Viết code, **chưa chạy được** trên máy dev | `selfhost/tts_server.py` + `selfhost/f5_tts_client.py` |

**Phát hiện quan trọng về ASR**: model Zipformer tiếng Việt công khai trên GitHub
release của k2-fsa thực ra là **non-streaming** (offline) — kiểm tra bằng
`onnx.load(...).metadata_props` thấy `comment = non-streaming zipformer2`, dù tên gọi
và mô tả trong roadmap là "streaming thật". Dùng `sherpa_onnx.OnlineRecognizer` (API
streaming) sẽ crash native vì thiếu metadata bắt buộc; phải dùng
`sherpa_onnx.OfflineRecognizer` (API batch). Hệ quả: `ZipformerVietnameseSTTService` là
`SegmentedSTTService` (transcribe trọn câu khi VAD báo hết lượt nói), không có partial
transcript giữa câu như một streaming ASR thật. Bù lại decode rất nhanh (0.09-0.11s cho
1.5-3.7s audio, đo thật trên CPU máy dev) nên không phải điểm nghẽn latency — chỉ là
mất khả năng "thấy" transcript trong lúc người dùng đang nói.

**Vì sao LLM/TTS chưa chạy được, không phải vì code sai**:
- **LLM**: vLLM chỉ có wheel `manylinux` trên PyPI, không có wheel Windows — phải chạy
  trong WSL2 (đã có `Ubuntu-24.04` trên máy) hoặc thuê GPU cloud Linux. Dù chạy được môi
  trường, GPU máy dev (2GB) không đủ VRAM cho Qwen3 8B kể cả lượng tử hoá AWQ (roadmap
  mục 10 khuyến nghị 1×A100/H100 hoặc 2×4090 cho khối này).
- **TTS**: checkpoint `model_last.pt` của F5-TTS-Vietnamese-ViVoice nặng ~5GB (kiểm tra
  qua HTTP HEAD trước khi tải, không tải bừa). F5-TTS là flow-matching (nhiều bước khử
  nhiễu mỗi lần sinh) — suy luận trên CPU (torch cài mặc định không có CUDA) sẽ rất
  chậm, GPU 2GB cũng khó đủ. Code (`selfhost/tts_server.py`) viết đúng theo API thật của
  `f5_tts.api.F5TTS` (đã đọc source, không đoán), nhưng chưa tải checkpoint + chạy thử
  đầu-cuối trong lần build này.
- **Vì sao TTS có venv riêng (`.venv-tts`)**: dependency của `f5-tts` (torch, gradio,
  hydra-core, wandb...) rất nặng và có nguy cơ ghi đè phiên bản `fastapi`/`starlette` mà
  Pipecat Gateway cần — đúng loại sự cố đã từng xảy ra với global Python env của máy này
  khi cài `pipecat-ai` (xem lịch sử: đã phải khôi phục `numpy`/`fastapi` về đúng phiên
  bản cũ). Rút kinh nghiệm, mọi dependency nặng/không chắc tương thích đều đi vào venv
  riêng, giao tiếp qua HTTP nội bộ (`selfhost/f5_tts_client.py`) thay vì cài chung.
- **Voice cloning cần consent**: F5-TTS-vi là zero-shot cloning, bắt buộc audio + text
  tham chiếu — cố tình KHÔNG có giọng mặc định dựng sẵn trong repo, người dùng phải tự
  cung cấp giọng mình có quyền sử dụng (`F5_TTS_REF_AUDIO`/`F5_TTS_REF_TEXT`), đúng yêu
  cầu "consent flow khi enroll giọng" ở `roadmap.md` mục 7.

Tải model ASR: `python scripts/download_asr_model.py` (chạy 1 lần, ~26MB, tải từ GitHub
release chính thức của k2-fsa/sherpa-onnx).

## Track A — Eval harness (đo lường trước khi đổi model)

Mục tiêu: có số liệu thật trước khi trả lời "model nhỏ hơn này có giảm chất lượng
nhiều không", thay vì đoán. 3 mảnh, tất cả nằm trong `eval/`:

| File | Đo gì | Trạng thái |
|---|---|---|
| `eval/asr_wer.py` | WER của `STT_BACKEND` đang cấu hình | **Đã chạy thật** trên local Zipformer: WER=0.000, latency trung bình 124ms (N=12 câu) |
| `eval/tool_call_accuracy.py` | LLM có gọi đúng tool (hoặc đúng không gọi) không | Đã verify plumbing đúng (context/tools/pipeline chạy tới tận API call) bằng key giả — cần `ANTHROPIC_API_KEY` thật để có kết quả |
| `eval/latency_log.py` + `eval/latency_report.py` | p50/p95 TTFA và turn latency, nhóm theo backend | Đã nối vào `bot.py`; cần chạy hội thoại voice thật để có dữ liệu |

**Phương pháp `asr_wer.py` — đọc kỹ trước khi tin số liệu**: dùng gTTS (Google Translate
TTS, miễn phí, không cần API key) tổng hợp audio từ 12 câu văn bản đã biết trước
(`eval/testset.py`), rồi đo WER của STT trên chính audio đó. Đây là **TTS round-trip
eval**, không phải benchmark trên giọng người thật (VLSP2020/2023, Common Voice,
FLEURS-vi) — những bộ đó cần tải qua kênh chính thức, chưa tải được trong lần build này
(đã thử `datasets-server` API của HuggingFace cho FLEURS-vi, bị chặn vì file parquet
vượt giới hạn quét; VIVOS trên HuggingFace cũng không truy vấn được qua preview API).
Audio TTS sạch hơn giọng người thật nhiều (không nhiễu, phát âm chuẩn) nên WER=0.000 là
**ideal case**, không phải WER thật ngoài đời — coi đây là smoke test hồi quy (so A vs
B cùng điều kiện) khi đổi model, không phải con số để công bố. Nâng cấp khuyến nghị:
thay bằng VIVOS/VLSP test set khi tải được.

Chạy: `python -m eval.asr_wer`, `python -m eval.tool_call_accuracy`,
`python -m eval.latency_report`.

## Track B — Provider-agnostic hoá 7 service

`services/anthropic_client.py` (hard-code Claude) đã đổi thành `services/llm_client.py`
— chuyển qua `SERVICES_LLM_BACKEND=cloud|local` giống mẫu `bot.py::build_llm`, nhưng là
switch riêng (7 service này và Speech Service là 2 pipeline độc lập, đổi model cho
voice không bắt buộc đổi luôn cho Vision/Planning/Reasoning/Generation/Text). `local`
trỏ vào bất kỳ endpoint tương thích OpenAI (vLLM) — cùng giới hạn chưa chạy được như
`LLM_BACKEND=local` của Speech Service (xem mục Giai đoạn 1 ở trên). Đã verify cả 2
backend construct đúng (Anthropic client + OpenAI-compatible client), chưa gọi API
thật cho backend `local` vì chưa có vLLM server chạy.

## Rủi ro đã biết

- **RAM**: máy dev chỉ ~2.1GB trống khi đo. Chạy đồng thời Docker + NATS + Ray + Gateway
  + trình duyệt có thể chậm. Đóng bớt ứng dụng khác khi test nếu thấy ì.
- **Ray trên Windows**: core actor (dùng trong dự án này) đã xác minh chạy tốt (Ray
  2.56.0, Python 3.10, `include_dashboard=False`). Nếu sau này gặp vấn đề với tính năng
  Ray nâng cao (placement groups, dashboard...), có thể thay `runtime/dispatcher.py`
  bằng một bản dùng `asyncio`/`ProcessPoolExecutor` thuần — interface `handle()` của
  từng service không đổi, chỉ phần "ai gọi handle()" đổi.
- **In-memory state**: `MemoryService` và `SessionManager` mất dữ liệu khi restart tiến
  trình. Chấp nhận được ở quy mô 1 dev; chuyển sang Redis khi cần bền vững hoặc nhiều
  instance Gateway/dispatcher.
