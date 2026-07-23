# Kế hoạch công việc, train và test model cho Speech-to-Speech tiếng Việt

> Cập nhật: 2026-07-20
> Phạm vi: nâng cấp pipeline `ASR -> Qwen3 -> TTS` hiện tại thành hệ thống
> **Dual-path Anchored Speech-to-Speech**, sau đó mới tiến tới audio-token S2S tiếng
> Việt thực sự.

### Trạng thái code hiện tại

Đã hoàn thành nền tảng, chưa train model mới:

- [x] In-process `AudioHub` và bounded shadow subscriber.
- [x] `SessionOrchestrator` quản lý turn/revision/generation/cancellation.
- [x] Typed segment, deterministic policy và tool/stale-revision hard gate.
- [x] Các mode `off|shadow|ack_only|speculative|primary`.
- [x] Interface `AudioTokenS2SBackend` và backend `probe` để test plumbing.
- [x] Tích hợp vào Pipecat mà giữ nguyên Anchor ASR-LLM-TTS.
- [x] Tích hợp Mimi codec thật; đã smoke-test 1 câu VIVOS và so sánh ASR trước/sau.
- [x] Adapter `moshi_ws` theo protocol Kyutai + resample/Opus/telemetry/fallback;
  integration test dùng mock server, chưa phải checkpoint.
- [ ] Tích hợp Moshi checkpoint thật.
- [ ] Thu thập benchmark audio người thật và chạy model go/no-go.

## 1. Quyết định kiến trúc

Không thay pipeline hiện tại ngay. Triển khai theo bốn mức:

```text
OFF
  -> SHADOW       : S2S chạy song song, không phát audio
  -> ACK_ONLY     : S2S/fast path chỉ phát backchannel và acknowledgement
  -> SPECULATIVE  : cho phép một số segment low-risk
  -> PRIMARY      : S2S là đường chính, cascade là fallback
```

Kiến trúc production mục tiêu:

```text
WebRTC -> AEC/NS -> In-process Audio Hub
                         |                     |
                         |                     +-> Anchor path
                         |                         Streaming ASR
                         |                         -> Qwen3
                         |                         -> RAG/Tool
                         |
                         +-> Interaction/S2S path
                             VAD/EoT/Backchannel
                             -> Moshi/Mimi

                 Cả hai path -> Session Orchestrator
                              -> Segment Policy/Verifier
                              -> Common TTS hoặc Mimi decoder
                              -> Output Commit Queue
                              -> WebRTC speaker
```

Ba nguyên tắc bắt buộc:

1. Audio frame không đi qua NATS hoặc Ray; NATS chỉ dùng cho control/tool/event.
2. Fast path không sở hữu số tiền, tên riêng, ngày giờ, trạng thái giao dịch hoặc kết
   quả tool.
3. Chỉ train model sau khi baseline, dữ liệu và tiêu chí go/no-go đã hoàn thành.

## 2. Bảng quyết định: model nào test, model nào train

| Khối           | Model/giải pháp                                                                    | Việc đầu tiên                                  | Có train ngay?         | Khi nào mới train                                                         |
| --------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------- |
| VAD             | Silero VAD hiện tại                                                                | Test mic thật, nhiễu, echo, overlap              | Không                  | Chỉ thay/fine-tune nếu false start hoặc missed interruption không đạt |
| Turn-taking     | `VietnameseBackchannelTurnStartStrategy` hiện tại                                | Test pause, backchannel, interruption              | Không                  | Train semantic EoT classifier nếu rule/VAD không đạt                    |
| ASR baseline    | Zipformer tiếng Việt hiện tại                                                    | Test WER trên audio mic thật và entity accuracy | Không                  | Chỉ fine-tune nếu lỗi domain còn cao sau hotword/text normalization     |
| ASR streaming   | Deepgram interim để kiểm chứng; candidate local streaming phải benchmark riêng | Test partial stability, final WER, latency         | Chưa                   | Fine-tune candidate thắng benchmark nếu domain/entity chưa đạt         |
| Anchor LLM      | Qwen3-8B-AWQ hiện tại                                                              | Test intent, tool calling, JSON schema, latency    | Không                  | LoRA/SFT khi prompt + schema + few-shot vẫn không đạt                   |
| Anchor LLM lớn | Qwen3-30B-A3B candidate                                                              | Chỉ benchmark ca reasoning/RAG khó               | Không                  | Không train trong MVP; chỉ dùng escalation nếu lợi ích đủ lớn      |
| RAG embedding   | Một embedding multilingual local, chọn bằng benchmark nội bộ                    | Test Recall@k trên tài liệu nghiệp vụ         | Không                  | Chỉ fine-tune contrastive nếu Recall@k không đạt                       |
| RAG reranker    | Một reranker multilingual local                                                     | Test nDCG/MRR và latency                          | Không                  | Chỉ fine-tune bằng hard negatives sau khi retrieval corpus ổn định     |
| TTS baseline    | VieNeu-TTS v3 Turbo hiện tại                                                       | Test TTFB, RTF, số/ngày/tên, long session       | Không                  | Chỉ fine-tune voice/style nếu chất lượng nội dung đã đạt          |
| Verifier        | Rule + entity comparator + tool-state gate                                           | Viết và test deterministic                       | Không có model        | Chỉ thêm NLI sau khi đo được rule không đủ                         |
| NLI verifier    | XLM-R/multilingual NLI candidate                                                     | Benchmark contradiction tiếng Việt               | Không                  | Fine-tune bằng cặp prefix-anchor thật nếu NLI generic không đạt      |
| Audio codec     | Mimi                                                                                 | Test encode/decode tiếng Việt 24 kHz             | Không                  | Chỉ fine-tune codec nếu làm sai thanh điệu/giảm intelligibility       |
| Duplex S2S      | Moshi                                                                                | Chạy inference và shadow benchmark               | Không ở vòng đầu   | LoRA/adaptation sau khi codec và data pilot vượt gate                    |
| Vietnamese S2S  | Moshi-VI dự kiến                                                                   | Train theo nhiều stage                            | Có, ở giai đoạn R&D | Chỉ sau khi có mono + stereo + event label đúng schema                  |
| Action channel  | Moshi-VI/kiến trúc kiểu DuplexSLA                                                 | Thiết kế schema và log event trước            | Không trong MVP        | Train sau khi S2S tiếng Việt và anchor injection đã ổn định         |

Tóm tắt ngắn:

- **Test, không train ngay:** Silero, Zipformer, Qwen3-8B, VieNeu-TTS, Mimi, Moshi.
- **Có thể fine-tune sau benchmark:** streaming ASR, embedding/reranker, NLI.
- **Chắc chắn cần adaptation để đạt đích audio-token tiếng Việt:** Moshi-VI.
- **Không fine-tune Mimi nếu codec tái tạo tiếng Việt đã tốt.**

## 3. Thứ tự công việc

## Giai đoạn 0 — Khóa baseline và bộ test

**Thời gian:** 1-2 tuần.
**Train model:** không.

### Công việc

- [ ] Tạo 300-500 câu test hội thoại tiếng Việt bằng audio người thật.
- [ ] Có giọng Bắc, Trung, Nam; nam/nữ; mic laptop, headset và điện thoại.
- [ ] Có noise văn phòng, đường phố, tiếng quạt và echo từ loa.
- [ ] Có pause, tự sửa câu, backchannel và chen lời.
- [ ] Có ít nhất 100 câu chứa số tiền, số điện thoại, ngày giờ, tên người, ngân hàng.
- [ ] Có 50-100 tình huống nhạy cảm: chuyển tiền, khóa thẻ, hủy giao dịch, OTP.
- [ ] Tách cố định `train/dev/test`; bộ `test` không được dùng để tạo synthetic train.
- [ ] Lưu audio gốc và metadata về thiết bị, vùng giọng, noise, SNR nếu có.

### Model cần test

1. Zipformer hiện tại:
   - WER tổng và WER theo vùng giọng.
   - Exact match cho `AMOUNT`, `DATETIME`, `PHONE`, `PERSON`, `BANK`.
   - Latency sau end-of-turn.
2. Qwen3-8B-AWQ:
   - Intent/sub-intent accuracy.
   - Tool-call name/argument exact match.
   - JSON/schema validity.
   - TTFT và total completion latency.
3. VieNeu-TTS:
   - TTFB và real-time factor.
   - Đọc đúng số, ngày, tiền, viết tắt và tên riêng.
   - Chất lượng sau 30-60 phút chạy liên tục.
4. Silero/backchannel strategy:
   - False interruption.
   - Missed interruption.
   - Pause bị nhận nhầm thành end-of-turn.

### Go/no-go

| Metric                               |                                   Gate ban đầu |
| ------------------------------------ | -----------------------------------------------: |
| ASR WER audio người thật          | <= 8% tổng; báo cáo riêng từng vùng giọng |
| Entity exact match cho ca nhạy cảm |                                           >= 98% |
| Tool call exact match                |     >= 98% và 0 side effect sai trong test gate |
| TTS GPU p50 TTFB                     |                                        <= 250 ms |
| TTS RTF                              |                                            < 0.7 |
| Interruption stop p50                |                                         < 200 ms |
| Sensitive factual error              |                                                0 |

**Deliverable:** báo cáo baseline JSON/Markdown và danh sách lỗi đã phân loại. Không
train model khi chưa có deliverable này.

## Giai đoạn 1 — Session Orchestrator và event schema

**Thời gian:** 1-2 tuần.
**Train model:** không.

### Công việc

- [ ] Tạo `SessionOrchestrator` sở hữu `session_id`, `turn_id`, `revision_id`,
  `generation_id` và `tool_execution_id`.
- [ ] Tạo cancellation token xuyên suốt ASR, LLM, RAG, tool, TTS và playback.
- [ ] Chỉ lưu assistant text tương ứng phần audio đã phát thật.
- [ ] Loại mọi output có `turn_id/revision_id` cũ.
- [ ] Tạo typed event schema thay vì truyền free-form text.
- [ ] Tạo output commit queue có playback acknowledgement.
- [ ] Test race condition: user chen lời đúng lúc tool/LLM/TTS trả kết quả.

Typed segment tối thiểu:

```json
{
  "turn_id": 24,
  "revision_id": 7,
  "segment_id": 3,
  "kind": "ack|social|fact|confirmation|transaction_result",
  "speech": "...",
  "risk": "low|medium|high",
  "claims": [],
  "requires_tools": [],
  "commit_policy": "immediate|anchor_verified|tool_verified"
}
```

### Go/no-go

- 100% audio pending của turn cũ bị hủy khi interruption.
- Không có tool result cũ đi vào turn mới.
- Conversation history khớp phần audio người dùng thực sự đã nghe.
- Audio frame không đi qua NATS/Ray.

## Giai đoạn 2 — Streaming ASR cho Anchor path

**Thời gian:** 2-3 tuần.
**Train model:** chưa; benchmark và tích hợp trước.

Zipformer hiện tại là segmented/offline ASR, nên không tạo partial transcript trong
lúc người dùng đang nói. Có thể tiếp tục dùng nó làm final transcript/fallback, nhưng
muốn prefetch RAG và intent sớm thì phải có streaming ASR thật.

### Công việc

- [ ] Dùng Deepgram interim làm reference để kiểm chứng orchestration streaming.
- [ ] Chọn 1-2 candidate ASR local streaming sau khi xác minh license/runtime.
- [ ] Benchmark trên cùng bộ audio Giai đoạn 0.
- [ ] Đo partial stability: token đã ổn định có bị sửa lại hay không.
- [ ] Thêm domain hotwords cho tên ngân hàng, sản phẩm và thuật ngữ nghiệp vụ.
- [ ] Chỉ cho prefetch tool read-only từ stable prefix.
- [ ] Giữ Zipformer để final rescore/fallback nếu candidate streaming kém hơn.

### Metric chọn model

```text
final WER
entity exact match
partial edit rate
stable-prefix delay
RTF
CPU/GPU/RAM per concurrent session
```

### Quyết định train

- Nếu candidate đạt gate: tích hợp, không train.
- Nếu WER đạt nhưng entity sai: thử hotword/context biasing trước.
- Nếu vẫn không đạt: LoRA/fine-tune ASR trên audio domain có transcript.
- Không train ASR bằng text synthetic không có audio tương ứng.

## Giai đoạn 3 — Anchor LLM, tool và RAG

**Thời gian:** 2-3 tuần.
**Train model:** mặc định không.

### Dữ liệu đang có được dùng ở đâu

Các file trong: LLLm

bao gồm `confirmation.accept`, `confirmation.update.*`, `confirmation.reject`, transfer,
payment, account, card, safety... phù hợp để:

- Test/train intent và sub-intent.
- Test/train entity extraction.
- Test dialogue state và pending action.
- Test tool selection/arguments.
- Tạo action-channel supervision.
- Làm kịch bản nguồn để render audio synthetic.

Các file này **chưa phải dữ liệu train Moshi/Mimi**, vì chưa có waveform, timestamp,
speaker channel, pause, overlap và interruption event.

### Công việc

- [ ] Chuyển taxonomy nghiệp vụ thành tool schema có version.
- [ ] Map `pending_action.action_id` sang idempotency key.
- [ ] Đánh nhãn tool thành `read_only`, `reversible_write`, `irreversible_write`,
  `sensitive`.
- [ ] Read-only được phép prefetch; write không được chạy speculative.
- [ ] Sensitive write bắt buộc user confirmation và tool success gate.
- [ ] Anchor LLM stream typed segments, không sinh hai output độc lập.
- [ ] Xây RAG evaluation set gồm query, relevant document và answer evidence.
- [ ] Benchmark embedding/reranker multilingual trước khi chọn model.

### Test Qwen3-8B trước khi nghĩ tới fine-tune

- Intent/sub-intent exact match.
- Tool name và arguments exact match.
- Không gọi tool khi không cần.
- Không phát `transaction_result` trước tool success.
- Không làm mất pending state ở `confirmation.accept/update/reject`.
- JSON schema validity.

### Khi nào fine-tune Qwen3

Chỉ LoRA/SFT nếu sau prompt/schema/few-shot vẫn xảy ra một trong các lỗi:

- Tool accuracy < 98%.
- Intent/sub-intent không đạt mục tiêu.
- Model thường xuyên phá schema.
- Model không tuân thủ confirmation/transaction policy.

Không train Qwen3 chỉ để giảm latency; latency phải giải quyết bằng serving, prompt,
cache, model routing và giới hạn output trước.

## Giai đoạn 4 — Segment Policy và Verifier deterministic

**Thời gian:** 1-2 tuần.
**Train model:** không.

### Policy ban đầu

```yaml
ack:
  allow_immediate: true
  buffer_ms: 0

social:
  allow_immediate: true
  buffer_ms: 80

fact:
  require_anchor: true
  buffer_ms: null

transaction_result:
  require_anchor: true
  require_tool_success: true
  buffer_ms: null
```

### Verifier theo thứ tự

1. Kiểm tra `turn_id/revision_id`.
2. Risk classification bằng rule/schema.
3. So khớp entity, số tiền, ngày, giờ, tên và phủ định.
4. Kiểm tra tool dependency/state.
5. Kiểm tra evidence/source ID cho factual claim.
6. Chỉ sau cùng mới dùng NLI nếu thực sự cần.

Output:

```text
ALLOW | WAIT | DROP | FALLBACK
```

### Khi nào test/train NLI

- Thu thập trước các cặp `(fast_prefix, anchor_segment, label)` từ shadow traffic.
- Benchmark multilingual NLI checkpoint trên tiếng Việt.
- Chỉ fine-tune khi có ít nhất vài nghìn cặp đã review và deterministic verifier không
  đủ giải quyết.
- NLI không được override entity/tool hard gate.

## Giai đoạn 5 — Fast acknowledgement và shadow S2S

**Thời gian:** 2-3 tuần.
**Train model:** chưa.

### Bước 5A — Fast path chưa dùng S2S sinh nội dung

- [ ] VAD/turn detector phát event `listen`, `pause`, `interrupt`, `backchannel`.
- [ ] Safe acknowledgement lấy từ danh sách kiểm soát.
- [ ] Toàn bộ audio dùng chung VieNeu-TTS để không đổi giọng.
- [ ] Ack không chứa entity hoặc factual claim.
- [ ] Anchor vẫn sở hữu nội dung chính.

### Bước 5B — Test Mimi

Pipeline:

```text
Vietnamese 24 kHz audio -> Mimi encode -> Mimi decode -> reconstructed audio
```

Test:

- Thanh điệu và dấu.
- Giọng Bắc/Trung/Nam.
- Số, tên người, địa phương, tên ngân hàng.
- Speaker similarity.
- Codec latency và RTF.
- WER của reconstructed audio so với audio gốc.
- Human MOS/AB test với audio gốc.

### Go/no-go cho Mimi

- ASR WER sau codec không tăng quá 2 điểm phần trăm tuyệt đối.
- Không có nhóm thanh điệu/vùng giọng bị lỗi hệ thống.
- Encode/decode chạy real-time ổn định.
- Nếu không đạt: fine-tune codec trước; chưa train Moshi-VI.

### Bước 5C — Test Moshi shadow

- [ ] Chạy checkpoint gốc, đo GPU memory, RTF và latency.
- [x] Fan-out audio từ Audio Hub sang adapter Moshi nhưng không phát output.
- [x] Log inner-monologue/prefix, first output, byte/frame và sequence-gap telemetry.
- [ ] So sánh turn-taking với rule/VAD hiện tại.
- [ ] Đo `prefix_accept_rate`, `prefix_contradiction_rate` và
  `anchor_ready_before_prefix_end`.

### Go/no-go

- Nếu prefix acceptance thấp: chỉ dùng Moshi cho interaction/ack.
- Nếu Moshi không chạy đồng thời với Qwen+TTS trên GB10: tách GPU hoặc chạy cloud;
  không giảm độ an toàn của anchor để nhường tài nguyên.

Smoke Mimi ngày 2026-07-20 trên ba speaker VIVOS người thật (`0000001`, `0003601`,
`0011360`): frame 80 ms, 8 codebook, cả ba có `WER delta = 0`. Sau 4 frame warm-up,
GPU codec RTF lần lượt 0,18 / 0,46 / 0,18 và first decoded frame 17-45 ms; cold warm-up
mất khoảng 6-10 giây. CPU ở mẫu đầu có RTF 7,97 và first decoded frame 334 ms. Kết quả
xác nhận runtime/wiring và khả năng realtime của Mimi trên GPU nhưng **không đủ** thông
qua codec gate; vẫn cần tập vùng miền/số/tên và MOS/AB test.

## Giai đoạn 6 — Chuẩn bị dữ liệu audio-token tiếng Việt

**Thời gian:** 3-6 tuần cho pilot; thu thập tiếp trong suốt dự án.
**Train model:** chưa, chỉ chuẩn hóa và kiểm định dữ liệu.

### 6A — Mono speech

Nguồn hiện đã có script hỗ trợ:

- VIVOS: dùng để test pipeline, quy mô nhỏ.
- viVoice và PhoAudiobook: cần quyền truy cập và kiểm tra license.
- Audio domain nội bộ có consent: ưu tiên cho giọng hội thoại ngân hàng.

Mục tiêu:

- Pilot: 100-500 giờ sạch để kiểm chứng training recipe.
- Scale: 5.000-20.000 giờ hoặc hơn nếu muốn language adaptation mạnh.

### 6B — Hội thoại stereo/dual-channel

Schema tối thiểu:

```json
{
  "audio_user": "...",
  "audio_assistant": "...",
  "sample_rate": 24000,
  "user_transcript": "...",
  "assistant_transcript": "...",
  "events": [
    {"time_ms": 1420, "type": "backchannel"},
    {"time_ms": 3180, "type": "user_interrupt"}
  ],
  "tool_events": [],
  "risk_labels": [],
  "dialect": "north|central|south",
  "consent": true
}
```

Mục tiêu:

- Pilot: 50-100 giờ stereo đã nghe/kiểm tra thủ công theo mẫu.
- MVP adaptation: 300-500 giờ stereo chất lượng cao.
- Synthetic: 500-2.000 giờ, nhiều giọng/noise/overlap.

### 6C — Chuyển text corpus hiện tại thành dialogue supervision

- [ ] Ghép `last_system_utterance`, user text và pending action thành multi-turn sample.
- [ ] Giữ `sample_id`, `semantic_family_id`, provenance và schema version.
- [ ] Thêm event `request_confirmation`, `accept`, `update`, `reject`, `tool_call`.
- [ ] Render nhiều giọng TTS để tạo pilot, không coi synthetic audio là test set.
- [ ] Sinh overlap/backchannel/interruption có timestamp thật.
- [ ] Review thủ công ít nhất 1-5% mỗi batch synthetic.

### Data gate

- Không có PII thật nếu chưa có consent và retention policy.
- Không dùng cùng speaker/semantic template giữa train và test nếu có nguy cơ leakage.
- Audio assistant và user không bị đảo channel.
- Transcript/timestamp/event phải qua validator tự động.
- Loại clipping, silence dài bất thường, TTS lỗi và mẫu entity đọc sai.

## Giai đoạn 7 — Train Moshi-VI theo nhiều stage

**Thời gian:** 6-12 tuần trở lên.
**Train model:** có.
**Tài nguyên:** GB10 phù hợp inference/experiment nhỏ; training nghiêm túc cần cụm GPU
cloud và phải benchmark memory/throughput trước khi thuê dài hạn.

Không train tất cả module cùng lúc.

### Stage 7.1 — Tokenizer và text/inner-monologue adaptation

- Adapt tokenizer cho tiếng Việt hoặc bổ sung vocabulary cần thiết.
- Train LoRA trên text/semantic stream trước.
- Freeze Mimi.
- Freeze phần lớn backbone ở pilot đầu.
- Đánh giá transcript, semantic content và language consistency.

### Stage 7.2 — User speech understanding

- Dạy user-audio stream -> delayed transcript/semantic stream.
- Dùng mono audio có transcript.
- Đánh giá ASR-like WER và intent/entity accuracy từ hidden/text stream.

### Stage 7.3 — Assistant Vietnamese speech

- Dạy semantic/text stream -> assistant audio token.
- Dùng audio assistant sạch, nhiều speaker và vùng giọng.
- Đánh giá intelligibility, tone, MOS, speaker consistency và RTF.

### Stage 7.4 — Full-duplex SFT

- Train trên stereo/dual-channel.
- Supervise pause, listen, speak, backchannel, interruption và end-turn.
- Trộn real stereo và synthetic, không dùng toàn synthetic.
- Đánh giá trên kịch bản overlap thật, không chỉ single-turn TTS.

### Stage 7.5 — Domain/action alignment

- Thêm action/control event từ corpus nghiệp vụ.
- Ban đầu chỉ train `ack`, `listen`, `wait`, `interrupt`, `tool_request`.
- Không cho model tự quyết irreversible tool.
- Tool execution vẫn qua Anchor/Session Orchestrator.

### Thứ tự freeze/unfreeze đề xuất

```text
Pilot 1: Mimi frozen + backbone frozen phần lớn + LoRA semantic/text
Pilot 2: Mimi frozen + LoRA temporal/depth layers liên quan speech
Pilot 3: unfreeze có chọn lọc nếu speech tiếng Việt chưa đạt
Codec:   chỉ unfreeze/fine-tune nếu Giai đoạn 5B chứng minh Mimi không đạt
```

### Eval sau mỗi checkpoint

- Offline Vietnamese speech quality.
- WER/entity accuracy từ speech sinh ra.
- Codec/audio RTF và first-audio latency.
- Pause/backchannel/interruption benchmark.
- Prefix factual contradiction.
- Safety và sensitive factual gate.
- Không promote checkpoint chỉ dựa trên training loss.

## Giai đoạn 8 — Relay handoff và speculative audio

**Thời gian:** 3-5 tuần.
**Train model:** có thể cần verifier/NLI nhẹ; không bắt buộc train backbone.

Triển khai theo thứ tự:

1. Text handoff.
2. Exact committed prefix.
3. Anchor continuation conditioned trên phần đã phát thật.
4. Clause-boundary handoff.
5. Common TTS renderer.
6. Sau khi ổn định mới thử Moshi audio -> anchor continuation.

Feature flags:

```env
S2S_MODE=off|shadow|ack_only|speculative|primary
S2S_COMMIT_BUFFER_MS=80
S2S_RISK_POLICY=anchored
S2S_ALLOW_FACTUAL_PREFIX=false
```

### Go/no-go để chuyển `ACK_ONLY -> SPECULATIVE`

- Prefix contradiction < 1% trên test gate.
- Sensitive factual error = 0.
- Prefix acceptance đủ cao để giảm latency có ý nghĩa.
- Không đổi giọng/prosody gây khó chịu ở điểm handoff.
- Interruption/cancellation không regression.
- First verified useful audio nhanh hơn baseline, không chỉ first sound.

Nếu không đạt, giữ `ACK_ONLY`; hệ thống vẫn có giá trị production và không bắt buộc
phải chuyển sang speculative factual speech.

## Giai đoạn 9 — Model-level anchor injection và action channel

**Thời gian:** R&D dài hạn.
**Train model:** có.

Chỉ bắt đầu sau khi Giai đoạn 7-8 đạt gate.

### Hướng nghiên cứu

- KAME-style: inject response/reference từ Anchor LLM vào S2S đang chạy.
- MoshiRAG-style: retrieval bất đồng bộ và reference stream.
- DuplexSLA-style: user audio, assistant audio và structured action cùng timeline.

### Điều kiện bắt đầu

- Moshi-VI đã nói/nghe tiếng Việt ổn định.
- Có log thật về thời điểm retrieval/tool result sẵn sàng.
- Có dataset action event gắn timestamp.
- Tool executor đã có idempotency và transaction policy.
- Có GPU budget cho ablation và retraining.

Không thay Session Orchestrator bằng action channel. Model đề xuất action; orchestrator
và policy vẫn là nơi quyết định thực thi.

## Giai đoạn 10 — Production hardening

**Thời gian:** 3-4 tuần trở lên.**Train model:** không, trừ khi canary phát hiện lỗi model có hệ thống.

- [ ] Distributed tracing theo session/turn/revision.
- [ ] GPU admission control và warm model pool.
- [ ] Load shedding và circuit breaker về cascade.
- [ ] Tool idempotency, timeout, retry và rollback policy.
- [ ] PII redaction, audio retention và consent policy.
- [ ] Canary theo `S2S_MODE`.
- [ ] Shadow comparison giữa checkpoint mới và production.
- [ ] Rollback model/config độc lập.
- [ ] Dashboard theo vùng giọng, loại noise và domain.

## 4. Ma trận test bắt buộc

| Test suite            | Zipformer/ASR | Qwen | VieNeu | Mimi |    Moshi-VI | Orchestrator |
| --------------------- | ------------: | ---: | -----: | ---: | ----------: | -----------: |
| Audio sạch 3 miền   |             x |      |      x |    x |           x |              |
| Noise/echo/far-field  |             x |      |        |    x |           x |            x |
| Số/tiền/ngày/tên  |             x |    x |      x |    x |           x |            x |
| Intent/sub-intent     |             x |    x |        |      |           x |            x |
| Confirmation state    |               |    x |        |      |           x |            x |
| Tool call/arguments   |               |    x |        |      | action only |            x |
| Pause/backchannel     |               |      |        |      |           x |            x |
| User interruption     |               |      |        |      |           x |            x |
| Stale revision/cancel |               |      |        |      |             |            x |
| Long session          |             x |    x |      x |    x |           x |            x |
| Concurrent sessions   |             x |    x |      x |    x |           x |            x |
| Sensitive safety gate |             x |    x |      x |      |           x |            x |

## 5. KPI cuối cùng

| Metric                           |      MVP | Production target |
| -------------------------------- | -------: | ----------------: |
| First acknowledgement p50        | < 400 ms |          < 250 ms |
| First verified factual audio p50 | < 900 ms |          < 600 ms |
| Interruption stop p50            | < 300 ms |          < 200 ms |
| False interruption               |    < 10% |              < 5% |
| Prefix contradiction             |     < 5% |              < 1% |
| Tool-call exact match            |   >= 98% |            >= 99% |
| Sensitive factual error          |        0 |                 0 |
| Cascade fallback rate            |    < 50% |          < 20-30% |
| Audio RTF                        |    < 0.7 |             < 0.5 |
| Task completion                  |    > 75% |             > 90% |

Metric chính để ra quyết định là:

```text
First Verified Useful Audio Latency
```

Không promote một model chỉ vì nó phát âm thanh sớm nếu nội dung phải bị sửa hoặc
fallback thường xuyên.

## 6. Lịch thực hiện rút gọn

| Tuần    | Công việc                                        | Train?                           |
| -------- | -------------------------------------------------- | -------------------------------- |
| 1-2      | Baseline + audio test set                          | Không                           |
| 2-4      | Session Orchestrator + cancellation + typed events | Không                           |
| 3-6      | Streaming ASR benchmark/integration                | Chưa; optional sau benchmark    |
| 5-7      | Anchor tool/RAG + deterministic verifier           | Không                           |
| 6-9      | Fast ack + Mimi/Moshi shadow benchmark             | Không                           |
| 7-12     | Chuẩn bị mono/stereo/synthetic audio data        | Không                           |
| 10-20+   | Moshi-VI staged adaptation                         | Có                              |
| 16-22+   | Relay handoff/speculative audio                    | Có thể cần verifier fine-tune |
| Sau đó | KAME/MoshiRAG/action-channel R&D                   | Có                              |

Các track data và infrastructure có thể chạy song song, nhưng thứ tự gate không được
đảo: **baseline -> orchestrator -> shadow -> data validation -> model training ->
speculative production**.

## 7. Việc nên bắt đầu ngay

1. Xây bộ test audio người thật từ Giai đoạn 0.
2. Thêm Session Orchestrator và typed segment schema.
3. Chuyển dữ liệu `Data_output/02_Hoang_Nhat` thành bộ test intent/tool/state chuẩn.
4. Benchmark ASR streaming thật; giữ Zipformer làm final/fallback.
5. Chạy Mimi resynthesis tiếng Việt trước khi tải/huấn luyện Moshi.
6. Chỉ sau năm bước trên mới quyết định GPU budget và recipe train Moshi-VI.

## 8. Tài liệu tham chiếu

- Kiến trúc hiện tại: [`platform-architecture.md`](platform-architecture.md)
- Roadmap tổng thể: [`roadmap.md`](roadmap.md)
- Moshi/Mimi: [https://github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- Moshi paper: [https://arxiv.org/abs/2410.00037](https://arxiv.org/abs/2410.00037)
- RelayS2S: [https://arxiv.org/abs/2603.23346](https://arxiv.org/abs/2603.23346)
- KAME: [https://arxiv.org/abs/2510.02327](https://arxiv.org/abs/2510.02327)
- MoshiRAG: [https://arxiv.org/abs/2604.12928](https://arxiv.org/abs/2604.12928)
- DuplexSLA: [https://arxiv.org/abs/2605.20755](https://arxiv.org/abs/2605.20755)
- J-Moshi language adaptation: [https://arxiv.org/abs/2506.02979](https://arxiv.org/abs/2506.02979)
