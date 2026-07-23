
# Kiến trúc nền tảng: Gateway / Event Bus / Runtime Scheduler / 9 service

Tài liệu này map từng ô trong sơ đồ kiến trúc gốc sang code/công nghệ thật trong repo,
giải thích các quyết định lệch so với đọc sơ đồ theo nghĩa đen, và đường mở rộng khi cần
scale vượt quá một máy dev.

## Bản đồ sơ đồ -> code

| Ô trong sơ đồ                                                 | Công nghệ                                                    | Code                                        |
| ----------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| Client SDK (WebRTC)                                               | Pipecat SmallWebRTC + prebuilt UI                              | `bot.py`, mount trong `gateway/main.py` |
| Client SDK (WebSocket)                                            | FastAPI WebSocket                                              | `gateway/main.py` (`/v1/ws`)            |
| Gateway → Authentication                                         | API key tĩnh qua header/query                                 | `gateway/auth.py`                         |
| Gateway → Session Manager                                        | dict in-memory                                                 | `gateway/session_manager.py`              |
| Gateway → Stream Multiplexer                                     | điều hướng theo`modality`                                | `gateway/multiplexer.py`                  |
| Gateway → Traffic Scheduler                                      | `asyncio.Semaphore` (per-session + toàn cục)               | `gateway/multiplexer.py`                  |
| Event Bus                                                         | NATS (JetStream), chạy qua Docker                             | `eventbus/client.py`                      |
| Runtime Scheduler                                                 | Ray, local mode (`ray.init()`)                               | `runtime/dispatcher.py`                   |
| Speech Service                                                    | pipeline Pipecat (Giai đoạn 0 cũ)                           | `bot.py`                                  |
| Vision / Text / Tool / Memory / Planning / Reasoning / Generation | Ray actor, bản mỏng gọi Claude                              | `services/*.py`                           |
| Monitoring Service                                                | subscriber thụ động trên Event Bus                         | `services/monitoring_service.py`          |
| GPU Cluster                                                       | **không có** — mọi cognitive service gọi Claude API | `services/anthropic_client.py`            |

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

| Khối                               | Model                                                              | Trạng thái                                                                                                 | Code                                                              |
| ----------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| STT                                 | Zipformer train ~70k giờ (zzasdf, qua sherpa-onnx) — mặc định | **Đã verify chạy** trên CPU, WER 2.7%, latency ~65ms/câu                                          | `selfhost/asr.py`                                               |
| STT (thay thế, không mặc định) | Whisper large-v3-turbo (ONNX qua sherpa-onnx)                      | **Đã đo, KHÔNG khuyến nghị**: WER 12.5% (thua Zipformer), latency ~2474ms/câu (~62x chậm hơn) | `selfhost/asr.py::WhisperTurboVietnameseSTTService`             |
| LLM                                 | Qwen3-8B-AWQ qua vLLM (API tương thích OpenAI)                  | **Đã verify chạy thật** trên GPU, tool-calling 100% (5/5 qua eval thật)                          | cấu hình trong`bot.py::build_llm`, venv riêng `.venv-vllm` |
| TTS                                 | VieNeu-TTS v3 Turbo (true streaming, venv riêng)                  | **Đã verify qua HTTP thật**: GPU warm p50 TTFB 207ms, p50 RTF 0.57; CPU p50 TTFB ~535ms, p50 RTF ~1.79 | `selfhost/tts_server.py` + `selfhost/vieneu_tts_client.py` |

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

**Sự cố thật đã gặp: ASR chất lượng kém trên giọng nói thật** — bản Zipformer-30M-RNNT
ban đầu (train 6.000 giờ) đo WER=3.5% trên benchmark tổng hợp (gTTS) nhưng người dùng
phản ánh chất lượng rất tệ trên hội thoại thật qua mic (xem log thật: câu nói bị nhận
dạng thành các câu tiếng Việt không liên quan). Nguyên nhân nhiều khả năng: benchmark
gTTS quá "sạch" (không nhiễu, phát âm chuẩn) nên không phản ánh đúng hiệu năng trên
giọng nói tự nhiên/có nhiễu — đúng giới hạn đã ghi trong `eval/testset.py`. Đã thử model
Zipformer khác **train trên ~70.000 giờ** (zzasdf/viet_iter3_pseudo_label, gấp ~10x dữ
liệu) — đo lại bằng `eval/asr_wer.py`, WER tổng giảm 3.5%→2.7% và code-switch 18.1%→14.3%
ngay cả trên benchmark cũ (chênh lệch thật trên giọng nói tự nhiên nhiều khả năng còn
lớn hơn nhiều so với con số benchmark). Đã đổi làm mặc định
(`scripts/download_asr_model.py`, không cần đổi code — chỉ đổi tên file sau khi giải nén
cho khớp quy ước cũ).

**Whisper large-v3-turbo đã thử làm lựa chọn ASR thứ ba, kết luận: không thay
Zipformer**. Cả hai chạy qua sherpa-onnx (không cần thêm runtime — Whisper turbo có
bản ONNX đã convert sẵn tại `csukuangfj/sherpa-onnx-whisper-turbo`). Đo bằng
`eval/asr_wer.py` trên tập categorized 63 câu (6 domain × 3 mức nhiễu), so với bản
Zipformer 70k giờ hiện là mặc định:

|                          | Zipformer (70k giờ, mặc định) | Whisper-turbo                             |
| ------------------------ | --------------------------------- | ----------------------------------------- |
| WER tổng                | 2.7%                              | 12.5%                                     |
| WER domain yếu nhất    | code_switch 14.3%                 | so_thoi_gian (số/giờ)**42.2%**    |
| WER code_switch VN-EN    | 14.3%                             | 15.2% (điểm mạnh duy nhất, sát nhau) |
| Latency trung bình/câu | 65ms                              | 2474ms (~38x chậm hơn)                  |

Whisper-turbo là model đa ngôn ngữ tổng quát, không train riêng tiếng Việt như
Zipformer (6000 giờ) — thua toàn diện ngoại trừ code-switching (cả hai đều chưa tốt ở
đó). Latency 2.5s/câu (CPU, chưa test GPU vì máy không có `onnxruntime-gpu`) không phù
hợp voice chat thời gian thực. Giữ lại làm lựa chọn bật được qua
`ASR_LOCAL_ENGINE=whisper` cho ai muốn tự thử, không đặt làm mặc định.

**LLM (Qwen3-8B-AWQ qua vLLM) — đã verify chạy thật, có một vấn đề phần cứng thật cần
biết**: GPU của máy test (NVIDIA GB10, kiến trúc Blackwell, compute capability sm_121)
quá mới — vLLM 0.25.1 chưa có kernel CUDA dựng sẵn cho sm_121 (xác nhận có issue mở trên
GitHub của vLLM về đúng vấn đề này trên phần cứng lớp DGX Spark/GB10), nên **lần chạy
đầu tiên phải tự biên dịch kernel, mất khoảng 15 phút** (torch.compile + FlashInfer
autotune + capture CUDA graph) — không phải bug, không phải treo, chỉ là chờ biên dịch.
Đã kiểm chứng: chat completion thật (tiếng Việt, tách đúng phần suy luận `<think>` khỏi
nội dung nhờ `--reasoning-parser qwen3`) và tool-calling thật qua `eval/tool_call_accuracy.py`
chạy trực tiếp qua `bot.py` — **100% (5/5)**.

Cài đặt (venv riêng `.venv-vllm`, xem `requirements-vllm.txt`):

```bash
python -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements-vllm.txt
.venv-vllm/bin/vllm serve Qwen/Qwen3-8B-AWQ \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --gpu-memory-utilization 0.15 --max-model-len 8192
```

Dùng bản lượng tử hoá **AWQ** (~6.1GB tải về, license Apache-2.0, repo
`Qwen/Qwen3-8B-AWQ`) thay vì bf16 gốc (~16GB) vì máy test dùng chung với nhiều người
khác, RAM/VRAM thực tế biến động theo thời điểm — **`--gpu-memory-utilization` phải nhỏ
hơn tỉ lệ bộ nhớ GPU thực sự còn trống lúc chạy** (vLLM tự báo lỗi rõ ràng nếu đặt quá
cao, vd `Free memory ... is less than desired GPU memory utilization`), kiểm tra bằng
`nvidia-smi`/`free -h` trước khi chọn số. `--tool-call-parser hermes` (không phải một
parser tên "qwen3") là giá trị đúng theo tài liệu triển khai chính thức của Qwen3.
Thiếu `--reasoning-parser qwen3` thì nội dung suy luận sẽ lẫn vào `content` — TTS sẽ đọc
to cả đoạn suy luận đó lên loa.

**Sự cố thật đã gặp: LLM trả lời rất chậm do "thinking mode" mặc định của Qwen3** —
qua log hội thoại thật, PROCESSING TIME mỗi lượt dao động 1.5-15.8 giây (một lượt tốn
1009 token prompt + **229 token completion** cho một câu trả lời đáng lẽ chỉ 1-3 câu)
dù TTFB rất nhanh (~0.1s) — toàn bộ thời gian bị "ăn" bởi phần suy luận `<think>` verbose
trước khi trả lời, đúng cơ chế mặc định của Qwen3. Khắc phục bằng
`--default-chat-template-kwargs '{"enable_thinking": false}'` ở cấp server (không cần
sửa `bot.py`/client code) — đã đo lại: completion token giảm còn ~60-70/lượt (giảm
~3-4 lần), **tool-calling vẫn đúng 100%** sau khi tắt thinking (có bug tương tác đã biết
giữa reasoning-parser + non-thinking mode ở các bản vLLM cũ hơn — 0.25.1 không gặp lại,
nhưng nên tự test lại tool-calling mỗi khi đổi phiên bản vLLM). Không dùng chuỗi
"/no_think" chèn vào prompt — đó chỉ là gợi ý mềm, không đảm bảo tắt hẳn như cờ server.

**Sự cố thật đã gặp khi chạy LLM local + TTS local cùng lúc trên máy ít VRAM**: TTS (VieNeu-TTS) mặc định
`VIENEU_DEVICE=auto` tự chọn GPU nếu có — nhưng khi vLLM đã chiếm phần lớn VRAM, TTS load
model sau **crash `CUDA error: out of memory`** ngay lúc có phiên hội thoại thật (im
lặng hoàn toàn, không có âm thanh trả lời, dù LLM đã trả lời đúng trong log). File mẫu
giữ `VIENEU_DEVICE=cpu` để tránh OOM. Tuy nhiên CPU trên máy GB10 hiện tại đo lại qua
endpoint streaming chỉ đạt p50 RTF ~1.79, có thể hụt audio ở câu dài. Nếu còn đủ bộ nhớ
sau khi vLLM load, dùng `VIENEU_DEVICE=cuda`; cấu hình local của máy này đã chuyển sang
CUDA và đạt p50 RTF 0.57.

**TTS**: dùng VieNeu-TTS (pnnbao97/VieNeu-TTS, gói pip `vieneu`, giấy phép Apache-2.0)
  thay cho F5-TTS-Vietnamese-ViVoice ban đầu (cc-by-nc-sa-4.0 — phi thương mại) — nhẹ hơn
  nhiều để cài (đường CPU không bắt buộc torch, dùng ONNX Runtime int8) và model tự tải
  về qua `huggingface_hub` ở lần chạy đầu (không cần script tải riêng như F5-TTS trước
  đây). Endpoint `/synthesize/stream` dùng `infer_stream()` và flush PCM16 từng chunk;
  endpoint WAV cũ vẫn giữ làm fallback. Đo qua đúng HTTP production với ba câu: CPU
  p50 TTFB ~535ms/RTF ~1.79; GPU sau warm-up p50 TTFB 207ms/RTF 0.57. Server warm-up
  một câu lúc khởi động để lượt hội thoại đầu không chịu cold-start/compile.

- **Vì sao TTS có venv riêng (`.venv-tts`)**: dù `vieneu` nhẹ hơn `f5-tts` nhiều, vẫn giữ
  venv riêng để cô lập khỏi Pipecat Gateway — đúng loại sự cố đã từng xảy ra với global
  Python env của máy này khi cài `pipecat-ai` (xem lịch sử: đã phải khôi phục
  `numpy`/`fastapi` về đúng phiên bản cũ). Giao tiếp qua HTTP nội bộ
  (`selfhost/vieneu_tts_client.py`) thay vì cài chung.
- **Voice cloning cần consent, nhưng có giọng dựng sẵn**: VieNeu-TTS đi kèm 14 giọng
  dựng sẵn (3 miền, nhiều phong cách) — chọn qua `VIENEU_VOICE` trong `.env`, KHÔNG bắt
  buộc tự cung cấp audio tham chiếu như F5-TTS trước đây. Vẫn hỗ trợ voice cloning zero-
  shot qua `VIENEU_REF_AUDIO` nếu muốn dùng giọng riêng — chỉ dùng audio bạn có quyền sử
  dụng (đã xin phép người nói), đúng yêu cầu "consent flow khi enroll giọng" ở
  `roadmap.md` mục 7. Audio sinh ra có watermark ẩn tích hợp sẵn (thư viện Perth).

Tải model ASR: `python scripts/download_asr_model.py` (chạy 1 lần, ~26MB, tải từ GitHub
release chính thức của k2-fsa/sherpa-onnx).
Model TTS (VieNeu-TTS) tự tải về qua `huggingface_hub` ở lần chạy `selfhost/tts_server.py`
đầu tiên — không cần script tải riêng.

## Track A — Eval harness (đo lường trước khi đổi model)

Mục tiêu: có số liệu thật trước khi trả lời "model nhỏ hơn này có giảm chất lượng
nhiều không", thay vì đoán. 3 mảnh, tất cả nằm trong `eval/`:

| File                                                 | Đo gì                                                    | Trạng thái                                                                                                                                       |
| ---------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval/asr_wer.py`                                  | WER của`STT_BACKEND` đang cấu hình                   | **Đã chạy thật** trên local Zipformer: WER=0.000, latency trung bình 124ms (N=12 câu)                                                 |
| `eval/tool_call_accuracy.py`                       | LLM có gọi đúng tool (hoặc đúng không gọi) không | Đã verify plumbing đúng (context/tools/pipeline chạy tới tận API call) bằng key giả — cần`ANTHROPIC_API_KEY` thật để có kết quả |
| `eval/latency_log.py` + `eval/latency_report.py` | p50/p95 TTFA và turn latency, nhóm theo backend          | Đã nối vào`bot.py`; cần chạy hội thoại voice thật để có dữ liệu                                                                    |
| `eval/tts_streaming_latency.py`                 | TTFB + RTF TTS qua HTTP streaming production             | **Đã chạy thật**: GPU p50 207ms / 0.57; có `--assert-realtime`                                                                     |

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
`python -m eval.latency_report`, `python -m eval.tts_streaming_latency --assert-realtime`.

## Track B — Provider-agnostic hoá 7 service

`services/anthropic_client.py` (hard-code Claude) đã đổi thành `services/llm_client.py`
— chuyển qua `SERVICES_LLM_BACKEND=cloud|local` giống mẫu `bot.py::build_llm`, nhưng là
switch riêng (7 service này và Speech Service là 2 pipeline độc lập, đổi model cho
voice không bắt buộc đổi luôn cho Vision/Planning/Reasoning/Generation/Text). `local`
trỏ vào bất kỳ endpoint tương thích OpenAI — dùng chung server vLLM đã verify chạy thật
ở `LLM_BACKEND=local` của Speech Service (xem mục Giai đoạn 1 ở trên) nếu muốn, qua
`SERVICES_VLLM_BASE_URL`. Đã verify cả 2 backend construct đúng (Anthropic client +
OpenAI-compatible client); riêng luồng service này (Vision/Text/Planning/Reasoning/
Generation) qua `local` chưa tự chạy thử gọi API thật, chỉ Speech Service
(`bot.py::build_llm`) đã verify đầy đủ.

## Track C — Giai đoạn 2: duplex (turn-taking, backchannel, phục hồi context, cảm xúc)

Code trong `duplex/`, nối vào `bot.py`. Ba phát hiện/quyết định đáng ghi lại:

**1. Semantic endpointing coi như đã xong từ Giai đoạn 0, không ai chủ ý làm.** Đọc
source `pipecat` 0.0.108 (`pipecat/turns/user_turn_strategies.py`) thấy
`UserTurnStrategies.__post_init__` mặc định dùng
`TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())` khi không truyền
`user_turn_strategies` — mà `bot.py` từ đầu chỉ truyền `vad_analyzer`, không truyền
`user_turn_strategies`, nên `LLMUserAggregator` tự rơi vào nhánh mặc định này (xem
`LLMUserAggregator.__init__`: `user_turn_strategies = self._params.user_turn_strategies or UserTurnStrategies()`). `LocalSmartTurnAnalyzerV3` bundle sẵn model ONNX
(`smart-turn-v3.2-cpu.onnx`), chạy CPU, không cần tải hay cấu hình gì thêm. Nói cách
khác: mục "nâng turn-taking từ VAD thuần lên semantic endpointing" của roadmap.md Giai
đoạn 2 coi như đã có sẵn miễn phí từ lúc dựng khung xương — chỉ là không ai biết/ghi lại
cho tới khi rà lại pipecat lần này.

**2. Backchannel filter dùng VAD có grace period thay vì bỏ tín hiệu VAD.**
`duplex/turn_strategies.py::VietnameseBackchannelTurnStartStrategy` thay `start`-strategy
mặc định, nhưng vẫn xử lý `VADUserStartedSpeakingFrame`. Khi bot đang nói, VAD mở timer
`DUPLEX_BARGE_IN_DELAY_MS` (mặc định 200ms): interim transcript "dạ/ừm" huỷ timer, câu
khác ngắt ngay, còn Zipformer local không có interim vẫn ngắt khi timer hết. Khi bot im
lặng, VAD mở turn tức thời. Unit test ba nhánh đã qua; benchmark classifier mô phỏng đạt
17/17, nhưng vẫn cần nghe/A-B bằng mic thật.

**3. `LLMContext.add_message(role="system", ...)` giữa hội thoại — hoạt động nhưng bị
biến đổi âm thầm với backend Anthropic.** `duplex/interruption_recovery.py` và
`duplex/emotion.py` đều chèn message `role="system"` vào giữa lịch sử hội thoại. Đọc
`pipecat/adapters/services/anthropic_adapter.py` xác nhận: Anthropic API không nhận
role "system" ngoài system prompt đầu tiên, nên adapter tự động đổi các message này
thành role "user" và gộp với message "user" liền kề (concat nội dung) trước khi gọi API
— không lỗi, nhưng LLM sẽ thấy nội dung đó như một phần lời người dùng nói (có tiền tố
`[Hệ thống: ...]`/`[user_emotion: ...]` để phân biệt), không phải một chỉ thị hệ thống
tách biệt thật sự. Backend OpenAI-compatible (`LLM_BACKEND=local`, vLLM) không có giới
hạn này — role "system" giữa hội thoại được giữ nguyên.

Cả 3 mảnh trong `duplex/` mới verify tới mức "plumbing đúng" (import thật + dựng
`Pipeline([...])` thật bằng `.venv` có cài `pipecat-ai==0.0.108`, key giả) — xem
README.md mục "Giai đoạn 2". Chưa nghe thử bằng hội thoại thật.

## Track D — Giai đoạn 3: chuẩn bị dữ liệu (`datagen/`)

`datagen/synthetic_dialogue.py` sinh hội thoại full-duplex 2 kênh tiếng Việt tổng hợp
(roadmap.md mục 8, điểm 1) — việc duy nhất của Giai đoạn 3 làm được mà không cần GPU và
không cần chốt trước lựa chọn (a)/(b)/(c) ở mục 2. Gọi ElevenLabs qua REST API trực tiếp
bằng `aiohttp` thay vì qua `pipecat.services.elevenlabs` — service đó thiết kế cho
streaming frame-based thời gian thực, không hợp để sinh file hàng loạt ở đây. Request/
response (endpoint `/stream/with-timestamps`, JSON-lines chứa `audio_base64`) sao chép
từ `HttpTTSService` thật trong `pipecat/services/elevenlabs/tts.py` (đọc source trong
`.venv`, không đoán field name).

Đã verify bằng test cục bộ (audio giả, không gọi mạng): logic ghép JSON kịch bản
(kể cả khi Claude bọc thêm `` ```json ``) và logic xếp audio 2 giọng vào buffer
stereo có chồng lấn thời gian thật ở lượt backchannel/interrupt (không chỉ nối chuỗi).
Chưa verify: gọi API thật (tốn phí Anthropic + ElevenLabs mỗi lần chạy) và chất lượng/
đa dạng của kịch bản Claude sinh ra có đủ tốt làm dữ liệu huấn luyện hay không — cần
chạy thử một mẻ nhỏ rồi nghe/đọc lại trước khi sinh số lượng lớn.

`datagen/talker_finetune_corpus.py` chuẩn bị corpus audio-text cho hướng (a) ở mục 2
roadmap (fine-tune Talker Qwen3-Omni nói tiếng Việt) — tải + resample 16kHz + ghi
manifest JSONL từ vivos/vivoice/phoaudiobook. ĐÃ CHẠY THẬT với vivos (công khai, xem
kết quả trong `data/talker_corpus/vivos/`): phát hiện `datasets` 5.x đã bỏ hỗ trợ
dataset loading script mà repo HF của vivos dùng (`vivos.py`, lỗi "Dataset scripts are
no longer supported") — sửa bằng cách tải thẳng `vivos.tar.gz` và tự parse
`prompts.txt`/cấu trúc thư mục thay vì gọi qua `datasets.load_dataset`. vivoice và
phoaudiobook ở dạng parquet hiện đại nên vẫn load bình thường qua `datasets`
streaming (chưa chạy thật — cả hai đều **bị gate trên HuggingFace**, yêu cầu tự xin
quyền bằng email trường/công ty tại trang dataset rồi `hf auth login`, xem docstring
đầu file để biết link chính xác).

**Phát hiện quan trọng khi chuẩn bị bắt tay vào Giai đoạn 3 (đo thật trên máy đang
chạy)**: roadmap.md mục 10 tự ước tính fine-tune Talker (kể cả chỉ LoRA) cần ~8×A100
80GB trong vài tuần. Máy hiện tại chỉ có 1×GPU GB10 dùng chung nhiều người — đo lúc
kiểm tra: GPU 93% bận và RAM hệ thống chỉ còn ~2GB/121GB trống, đều do tiến trình của
user khác (`tts01`, `tts02` train object detection/ablation, `ai02` chạy OCR service).
Tức là bước train thật của Giai đoạn 3 không khả thi trên máy này ở quy mô roadmap đã
ghi — quyết định cùng người dùng: làm phần chuẩn bị dữ liệu ngay (không cần GPU), để
ngỏ quyết định thuê cloud GPU cho bước train đến khi có đủ dữ liệu.

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
- **Backchannel filter chưa tune bằng dữ liệu thật**: danh sách từ trong
  `duplex/turn_strategies.py::DEFAULT_BACKCHANNEL_WORDS` là khởi điểm chủ quan (dựa trên
  ví dụ trong roadmap.md), chưa đối chiếu với hội thoại tiếng Việt thật — có thể vẫn lọt
  sót biến thể vùng miền hoặc lọc nhầm câu trả lời ngắn hợp lệ.

## Track E — Dual-path S2S control/media plane

Đã triển khai nền runtime trong `s2s/` mà không thay đường Anchor đã verify:

```text
transport.input
  -> AudioHubProcessor -----------------------> AudioHub -> ShadowS2SRunner
  -> STT -> user context -> RealtimeControl   <- fast proposal + SegmentPolicy
  -> LLM -> AnchorObservation -> TTS
  -> PlaybackState -> transport.output
```

Quyền sở hữu state:

- `SessionOrchestrator`: `turn_id`, `revision_id`, `generation_id`, cancellation,
  segment decision và audio queued/played estimate.
- `AudioHub`: PCM16 ring buffer và bounded queue cho mỗi model consumer. Queue fast
  path đầy thì drop frame cũ của fast path; không chặn media producer.
- `SegmentPolicy`: `ALLOW|WAIT|DROP|FALLBACK`; fast factual/tool content không được
  commit kể cả khi `S2S_MODE=primary`.
- `AudioTokenS2SBackend`: contract model-neutral. `probe` đếm frame; `moshi_ws` kết nối
  sidecar chính thức qua WebSocket, resample PCM16 sang Opus mono 24 kHz, quan sát
  audio/inner-monologue nhưng không commit model audio. URL ngoài loopback bị chặn mặc
  định và lỗi sidecar không chặn Anchor.

Các mode áp dụng theo phiên mới:

| Mode | Hành vi |
|---|---|
| `off` | Không khởi động consumer fast path |
| `shadow` | Fan-out audio, log telemetry, không phát fast output |
| `ack_only` | Chỉ commit ACK/backchannel không factual qua common TTS |
| `speculative` | Cho phép low-risk proposal; facts/tools vẫn chờ anchor |
| `primary` | Ưu tiên S2S renderer tương lai nhưng không bypass policy |

Dashboard `/api/status` trả thêm snapshot S2S không chứa transcript/audio. Settings có
thể đổi `S2S_MODE` cho phiên voice mới. Unit tests cho cancellation, stale revision,
tool gate, factual-prefix gate và AudioHub overflow nằm trong
`tests/test_s2s_control_plane.py`.
