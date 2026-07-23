# Lộ trình Speech2Speech tiếng Việt

> Bản markdown dễ đọc/dễ grep của `speech2speech.pdf` (trích xuất bằng `pdftotext`,
> dọn lại bảng biểu). Nếu có sai lệch, file PDF gốc là nguồn chính xác.

> **Pivot kiến trúc (2026-07-17)**: Giai đoạn 2 (full duplex) và Giai đoạn 3 (speech
> token/Talker fine-tune) ở mục 2 dưới đây đã được thay bằng kiến trúc **Dual-path
> Anchored Speech-to-Speech** chi tiết hơn — xem [mục 11](#11-pivot-kiến-trúc-quyết-định-2026-07-17-dual-path-anchored-speech-to-speech).
> Giai đoạn 0-1 (mục 2) vẫn giữ nguyên, đã verify chạy thật, và trở thành "Cascade
> fallback" trong kiến trúc mới — không có gì bị bỏ.

Ý tưởng cốt lõi của sơ đồ: mọi thứ đều streaming theo chunk, LLM anchor là nơi duy nhất
"hiểu" — ASR/TTS chỉ là bộ chuyển đổi, kênh cảm xúc chạy song song để không mất
paralinguistic khi audio bị nén thành text, và đường ngắt lời (barge-in) cắt thẳng TTS
mà không cần đi qua LLM.

## 1. Ba hướng kiến trúc và lý do chọn hybrid

| Hướng | Ưu | Nhược | Vai trò trong plan |
|---|---|---|---|
| A. Cascaded streaming (ASR→LLM→TTS) | Tool calling native, tiếng Việt tốt ngay, mỗi khối thay được | Mất prosody qua text, latency ~700–900ms, turn-taking cứng | Giai đoạn 1–2 (nền tảng) |
| B. End-to-end speech-native (Moshi-style) | Full duplex thật, latency thấp nhất | Moshi đạt ~200ms nhưng giới hạn tiếng Anh và khả năng suy luận cỡ 7B, Inworld tool calling yếu, không có tiếng Việt | Tham chiếu nghiên cứu |
| C. Hybrid LLM-anchor (Thinker–Talker / micro-turn) | Giữ trí tuệ + tools của text LLM, speech tokens giữ cảm xúc, streaming từng frame | Cần train Talker cho tiếng Việt | Giai đoạn 3–4 (đích đến) |

Hai bằng chứng ủng hộ lộ trình này: một tutorial kỹ thuật tháng 3/2026 đo được pipeline
cascaded tự host đạt ~755ms time-to-first-audio, trong khi Qwen3-Omni tự host chỉ chạy
tốt phần Thinker qua vLLM (516ms audio-to-text) còn Talker tối ưu chỉ có trên cloud API
(~702ms) — tức là cascaded và omni-model hiện gần như hòa nhau về latency, nhưng cascaded
cho quyền kiểm soát và tiếng Việt. Đồng thời kiến trúc Thinker–Talker của Qwen3-Omni cố
tình tách rời hai phần để hệ thống bên ngoài (RAG, safety filter, function call) có thể
can thiệp giữa bước sinh text và bước sinh speech — đây chính là mẫu thiết kế "LLM
anchor" nên bám theo, kể cả khi tự build.

## 2. Lộ trình 4 giai đoạn

**Giai đoạn 0 — Khung xương (1–2 tuần).** Dựng transport WebRTC + framework agent
(Pipecat hoặc LiveKit Agents — cả hai đều có sẵn plugin ASR/LLM/TTS, interruption
handling, và smart turn detection). Chạy baseline bằng API thương mại
(Deepgram/Whisper API + Claude/GPT + ElevenLabs) để chốt UX, đo latency từng chặng, viết
sẵn eval harness. Đừng bỏ qua bước này: nó cho thước đo trước khi thay từng khối bằng
model tự host. *(→ đây là phần đã dựng trong repo này, xem `README.md`.)*

**Giai đoạn 1 — Cascaded tự host cho tiếng Việt (3–6 tuần).** Thay lần lượt: ASR bằng
Zipformer streaming tiếng Việt, LLM bằng Qwen3 serve qua vLLM (streaming token + function
calling), TTS bằng một model tiếng Việt hỗ trợ streaming theo câu/chunk. Barge-in phiên
bản 1 dựa trên VAD: người dùng nói ≥200–300ms trong lúc bot đang phát → flush audio
buffer, huỷ request TTS/LLM, ghi lại "assistant bị ngắt tại từ thứ N" vào context. Mục
tiêu: TTFA < 1s, tool calling hoạt động.

*Trạng thái triển khai 2026-07-20:* WebRTC/LLM/TTS đã streaming; VieNeu-TTS GPU đo p50
first audio 207ms, RTF 0.57. Zipformer checkpoint đang dùng thực tế là offline decode
theo lượt nói (dù decode nhanh ~65ms), nên đây là cascaded real-time chứ chưa phải mọi
khối đều streaming frame-level.

**Giai đoạn 2 — Full duplex "thật" + cảm xúc (4–6 tuần).** Nâng turn-taking từ VAD thuần
lên semantic endpointing (model smart-turn của Pipecat hoặc turn detector của LiveKit, có
thể fine-tune thêm tiếng Việt); phân biệt backchannel ("ừ", "vâng vâng") với ngắt lời
thật. Thêm kênh cảm xúc: SER trên audio đầu vào → chèn thẻ `[user_emotion: frustrated]`
vào context LLM → LLM sinh thẻ điều khiển giọng → TTS instruct. Hướng nghiên cứu đáng
bám: DuplexCascade bỏ hẳn VAD, chuyển hội thoại thành micro-turn theo chunk và huấn luyện
LLM với các control token đặc biệt để tự quyết khi nào nói/nghe/im lặng — đạt SOTA
turn-taking full-duplex trong nhóm mã nguồn mở mà vẫn giữ trí tuệ của text LLM. Có thể
tái hiện cách này trên Qwen3 với dữ liệu hội thoại tiếng Việt.

**Giai đoạn 3 — Speech tokens với LLM anchor (2–3 tháng).** Thay cặp ASR/TTS bằng đường
audio-token trực tiếp nhưng vẫn giữ text làm anchor. Ba lựa chọn xếp theo độ rủi ro tăng
dần:
- (a) fine-tune Talker của Qwen3-Omni cho tiếng Việt — model này đã hiểu speech tiếng
  Việt ở đầu vào (1 trong 19 ngôn ngữ speech input) nhưng chỉ sinh speech ra 10 ngôn ngữ,
  chưa có tiếng Việt, nên đóng góp là dạy Talker nói tiếng Việt bằng
  viVoice/PhoAudiobook theo quy trình huấn luyện Talker 4 giai đoạn mô tả trong technical
  report;
- (b) theo dõi Qwen3.5-Omni — bản công bố tháng 3/2026 tuyên bố sinh speech 36 ngôn
  ngữ, có semantic interruption, voice cloning và điều khiển cảm xúc, nhưng việc mở
  weights lúc đó chưa được xác nhận (đáng search lại trước khi bắt tay);
- (c) adapt Moshi sang tiếng Việt kiểu J-Moshi đã làm với tiếng Nhật — cần dữ liệu hội
  thoại 2 kênh, khó nhất nhưng full-duplex thuần nhất.

**Giai đoạn 4 — Voice cloning + cá nhân hóa (song song cuối giai đoạn 3).** Zero-shot
cloning từ 3–10 giây audio tham chiếu, kèm flow xin consent và audio watermarking.

## 3. Model dùng được ngay cho từng khối

| Khối | Cho tiếng Việt | Đa ngôn ngữ / thay thế |
|---|---|---|
| ASR streaming | Mục tiêu là Zipformer online; checkpoint Zipformer 70k giờ hiện tích hợp qua sherpa-onnx là **offline theo lượt** nhưng decode nhanh ~65ms; PhoWhisper cho rescoring | Deepgram Nova-3 là fallback streaming thật; Whisper large-v3-turbo local đã đo quá chậm (~2.5s/câu) |
| VAD + turn | Silero VAD v5 hoặc TEN VAD; smart-turn v3 (Pipecat) / LiveKit turn detector, fine-tune thêm tiếng Việt | Semantic endpointing bằng LLM nhỏ (Qwen3-0.6B) đọc partial transcript |
| LLM anchor | Qwen3 8B–32B (tiếng Việt mạnh, function calling tốt, vLLM streaming); thay thế: Llama 3.3, SeaLLM/Sailor cho Đông Nam Á | Claude/GPT qua API nếu ưu tiên chất lượng tool-use |
| TTS streaming | VieNeu-TTS v3 Turbo hiện dùng `infer_stream()` thật (GPU p50 first audio 207ms, RTF 0.57); F5-TTS/viXTTS/VietTTS là lựa chọn cần benchmark thêm | CosyVoice 2/3 (streaming 2 chiều, emotion instruct); thương mại: ElevenLabs `eleven_flash_v2_5` |
| Emotion-in (SER) | emotion2vec+ large đóng băng + classifier nhẹ — đúng công thức các đội VLSP 2025 dùng cho SER tiếng Việt; SenseVoice (ASR + emotion + audio event) | Qwen3-Omni-Captioner để mô tả paralinguistic chi tiết |
| E2E speech LLM (GĐ3) | Qwen3-Omni-30B-A3B (input tiếng Việt ✓, output cần train) | Moshi, Freeze-Omni, GLM-4-Voice; PersonaPlex (NVIDIA, 2026) — full duplex có điều khiển giọng và vai |

## 4. Full duplex và barge-in — chi tiết kỹ thuật

Bốn tầng phải làm đúng thứ tự:

1. **Echo cancellation.** Bot phát ra loa thì mic thu lại chính giọng bot — không có AEC
   tốt (WebRTC AEC3 phía client, hoặc reference-signal subtraction phía server) thì mọi
   logic barge-in đều vô nghĩa vì bot tự ngắt lời mình.
2. **Dual-state VAD.** Luôn chạy VAD trên luồng mic kể cả khi bot đang nói; trạng thái hệ
   thống là máy trạng thái `{listening, thinking, speaking, interrupted}`.
3. **Phân loại ngắt lời.** 200ms tiếng nói khi bot đang phát chưa chắc là barge-in — có
   thể là backchannel hoặc ho; chạy partial ASR + classifier nhỏ trên 300–500ms đầu để
   quyết định cắt hay tiếp tục (đáng fine-tune cho tiếng Việt vì backchannel "dạ", "ừm",
   "vâng" rất đặc trưng).
4. **Phục hồi context.** Khi cắt TTS phải biết bot đã phát đến từ nào để ghi vào lịch sử
   hội thoại dạng `"assistant: ... (bị ngắt)"`, nếu không LLM sẽ tưởng người dùng đã nghe
   hết.

Ngân sách độ trễ mục tiêu cho cascaded (P50): AEC+VAD ~20ms, ASR partial cuối
~150–250ms sau khi ngừng nói, endpointing ~100–200ms, LLM first token ~150–300ms (vLLM,
model 8B), TTS first chunk ~100–200ms, mạng WebRTC ~50ms — tổng ~600–950ms, khớp với con
số thực nghiệm đã công bố. Barge-in phải nhanh hơn nhiều: từ lúc phát hiện tiếng nói đến
lúc loa im < 150ms, vì đường này không đi qua LLM.

## 5. Tool calling khi đang streaming

Trong cascaded, đây là điểm mạnh nhất của LLM anchor: LLM stream text, khi gặp tool call
thì framework tạm dừng TTS, chạy tool async, rồi tiếp tục. Ba mẹo thực chiến:

1. Cho LLM sinh một câu "đệm" nói được ngay trước khi gọi tool ("Để mình kiểm tra lịch
   nhé...") để lấp khoảng chờ.
2. Tool chậm >2s thì phát filler audio hoặc earcon.
3. Parse tool call theo kiểu incremental trên stream chứ đừng đợi hết response.

Sang giai đoạn 3, Qwen3-Omni đã hỗ trợ function calling native qua thẻ `tool_call` nên
kiến trúc này chuyển tiếp mượt.

## 6. Emotion preservation

Vấn đề gốc: text là nút cổ chai làm mất prosody. Giải pháp theo giai đoạn — ở giai đoạn
2, đi vòng qua nút cổ chai bằng metadata: SER trích nhãn cảm xúc + đặc trưng (tốc độ nói,
cao độ, cười) → chèn vào context LLM dạng thẻ → LLM quyết định giọng đáp và sinh thẻ điều
khiển (`<style: warm, slow>`) → TTS instruct render. Ở giai đoạn 3, giải quyết tận gốc:
Talker sinh speech từ multi-codebook token chứa thông tin âm học, và Qwen3-Omni cố ý cho
Talker điều kiện hóa trực tiếp trên đặc trưng audio đa phương thức thay vì chỉ text,
chính là để bảo toàn prosody và âm sắc.

Lưu ý dữ liệu: SER tiếng Việt rất mỏng — ViSEC chỉ có 5.400 câu, các đội VLSP phải trộn
thêm IEMOCAP, RAVDESS, CREMA-D, EmoV-DB — nên chiến lược thực tế là dùng SER
cross-lingual (emotion2vec) + tự tạo 10–20 giờ dữ liệu cảm xúc tiếng Việt bằng diễn viên
hoặc bằng TTS instruct để augment.

## 7. Voice cloning (giai đoạn sau)

Đường ngắn nhất: viXTTS hoặc VietTTS đã có zero-shot cloning tiếng Việt sẵn; F5-TTS-vi
cũng clone zero-shot từ audio tham chiếu. Nếu muốn chất lượng cao hơn, fine-tune
CosyVoice 3 hoặc F5 trên PhoAudiobook — bộ này có speaker ID riêng cho từng mẫu (735
người đọc), điều mà viVoice thiếu vì chỉ dùng tên kênh YouTube làm proxy, nên hợp cho
speaker-conditioned training. Bắt buộc kèm: consent flow khi enroll giọng, watermark
audio đầu ra, và test speaker-similarity bằng ECAPA-TDNN cosine trên tập Vietnam-Celeb.

## 8. Dữ liệu tiếng Việt

| Bộ dữ liệu | Quy mô | Loại | Dùng cho |
|---|---|---|---|
| PhoAudiobook (Movian AI, ACL 2025) | 941h, 735 speaker | Audiobook, text đã chuẩn hóa, có speaker ID | TTS, cloning, train Talker |
| viVoice (capleaf) | 1.000h+, 186 kênh YouTube | Đa dạng, đã lọc nhiễu, 24kHz | TTS, Talker; lưu ý text chưa chuẩn hóa |
| Bud500 (VietAI) | ~500h | Podcast, du lịch, sách; đủ giọng 3 miền | ASR |
| VinBigData VLSP2020 | ~100h | Đọc + hội thoại | ASR fine-tune, test chuẩn |
| VietSuperSpeech (2026) | ~240h | Hội thoại tự nhiên kiểu call center, pseudo-label | ASR hội thoại — đúng register cho voice agent |
| VIVOS / FLEURS-vi / Common Voice vi | 15h / ~10h / ~17h | Đọc | Test set, đánh giá |
| GigaSpeech2-vi / VietMed / ViMD | vài trăm–nghìn giờ / 16h / 100h | Crawl YouTube / y tế / phương ngữ | Pretrain thêm, domain |
| ViSEC | 5.400 câu | SER tiếng Việt | Emotion-in (kết hợp cross-lingual) |
| Vietnam-Celeb | ~1.000 speaker | Xác thực người nói | Đánh giá cloning |

Dữ liệu sẽ phải tự tạo (không có sẵn công khai):

1. Hội thoại full-duplex 2 kênh tiếng Việt cho giai đoạn 3 — cách làm giống Moshi: crawl
   podcast/talkshow rồi tách kênh bằng diarization, cộng với sinh hội thoại tổng hợp (LLM
   viết kịch bản có ngắt lời, backchannel → TTS đa giọng render thành 2 stream).
2. Speech-instruction tiếng Việt — lấy bộ instruct text tiếng Việt sẵn có, TTS hóa phần
   câu hỏi bằng nhiều giọng/nhiễu để train khả năng nghe-hiểu-lệnh.
3. ~10–20h cảm xúc có kịch bản để đánh giá emotion preservation.

## 9. Đánh giá và các hướng ra paper

Metrics: TTFA và barge-in latency (P50/P95); WER trên VLSP2020/2023 test; MOS/SMOS và
UTMOS cho TTS; speaker similarity cho cloning; emotion accuracy vào–ra; tool-call
accuracy; và quan trọng nhất — Full-Duplex-Bench (v1 và v1.5 cho overlap handling) để đo
turn-taking.

Ba hướng research-friendly có khoảng trống rõ:

1. Talker tiếng Việt cho Qwen3-Omni — chưa ai công bố speech output tiếng Việt cho họ
   model này.
2. Tái hiện micro-turn control tokens của DuplexCascade trên dữ liệu hội thoại tiếng
   Việt.
3. Benchmark full-duplex đầu tiên cho tiếng Việt (backchannel "dạ/vâng" là hiện tượng
   ngôn ngữ học thú vị riêng).

## 10. Phần cứng

Giai đoạn 1–2 chạy được trên 1×RTX 4090/A10 cho ASR+TTS cộng 1×A100/H100 (hoặc 2×4090)
cho LLM 8B–32B lượng tử hóa AWQ qua vLLM; mỗi phiên hội thoại đồng thời tốn thêm ít vì
các model streaming đều nhẹ. Giai đoạn 3 fine-tune Talker/omni-model cần cỡ 8×A100 80GB
trong vài tuần với LoRA trên phần Talker, hoặc thuê cloud theo đợt.

**Máy dev hiện tại (GPU 2GB VRAM) không đủ cho giai đoạn 1 trở đi** — mọi việc tự host
model đều cần thuê GPU cloud theo đợt.

## 11. Pivot kiến trúc (quyết định 2026-07-17): Dual-path Anchored Speech-to-Speech

> **Nguồn**: nội dung mục này tổng hợp từ đề xuất kiến trúc do người dùng cung cấp trực
> tiếp trong hội thoại, không phải do tự nghiên cứu/fetch mạng. Các claim về model cụ
> thể (Moshi/Mimi, MoshiRAG, KAME, DuplexSLA, RelayS2S, Qwen3-ASR-0.6B/1.7B, MiniCPM-o
> 4.5, GLM-4-Voice, Kani TTS Vie, Valtec TTS) nằm ngoài khả năng tự verify tại thời điểm
> ghi (kiến thức tới 2026-01, không có bước fetch/tra cứu độc lập) — coi là **giả thuyết
> cần tự kiểm chứng license/tồn tại thật/chất lượng** trước khi cam kết từng track, đúng
> tinh thần "đọc source thật, không đoán" đã áp dụng xuyên suốt tài liệu này (xem cách
> Whisper-turbo, F5-TTS, Zipformer-6k-giờ đã được tự đo lại thay vì tin số liệu công bố).

### Vì sao pivot khỏi Giai đoạn 2-3 cũ ở mục 2

Giai đoạn 3 cũ (hướng (a) đã chọn) đặt cược vào việc tự fine-tune Talker của Qwen3-Omni
nói tiếng Việt trước khi có bất kỳ demo full pipeline nào chạy được — rủi ro cao, không
có gate trung gian. Vấn đề gốc: **mọi model S2S native mã nguồn mở hiện tại (Moshi,
Qwen3-Omni, MiniCPM-o, GLM-4-Voice) đều chưa có checkpoint chính thức sinh speech tiếng
Việt production-ready** (Qwen3-Omni hiểu tiếng Việt ở input nhưng chỉ sinh speech ra 10
ngôn ngữ, chưa gồm tiếng Việt — xem mục 2 hướng (a) đã ghi).

Dual-path Anchored tách hai trách nhiệm thay vì gộp vào một model:

- **Fast path** (S2S) — chỉ chịu trách nhiệm turn-taking/backchannel/ack/prosody, KHÔNG
  chịu trách nhiệm nội dung factual.
- **Anchor path** (ASR + text LLM) — chính là Giai đoạn 1 đã verify chạy thật
  (Zipformer + Qwen3-8B-AWQ), chịu trách nhiệm toàn bộ nội dung chính xác.
- **Verifier** — quyết định phần nào của fast path được phép phát ra loa (COMMIT/HOLD/
  REWRITE/CANCEL/CASCADE_FALLBACK).
- **Cascade fallback** — chính là pipeline cascaded hiện tại
  (`bot.py::build_stt/build_llm/build_tts`), dùng khi rủi ro cao (thanh toán, y tế,
  pháp lý...) hoặc khi fast path và anchor bất đồng.

**Điểm quan trọng nhất: không có gì trong Giai đoạn 0-1 bị bỏ.** Cascaded pipeline đã
verify trở thành Anchor path + Cascade fallback trong kiến trúc mới — nguyên xi code
hiện tại, không viết lại. Lợi ích của pivot: không cần Vietnamese Talker sẵn sàng ngay
để có một hệ thống chạy được — fast path S2S có thể chạy ở "shadow mode" (không phát
audio thật) trong khi anchor path (100% code đã verify) vẫn chịu trách nhiệm toàn bộ
nội dung, giống hệt cấu hình A đề xuất bên dưới.

### Kiến trúc tóm tắt

```
Mic → AEC/NS → Shared audio bus (ring buffer, timestamp, turn_id)
                    │                              │
             FAST PATH (S2S)                 ANCHOR PATH
        turn-taking/backchannel/         Streaming ASR → Anchor LLM
        speculative low-risk prefix      (reasoning/RAG/tool/factual)
                    │                              │
                    └──────────► VERIFIER ◄─────────┘
                    risk classify · entity/number check
                    semantic entailment · tool-state gate
                                │
                ┌───────────────┼───────────────┐
             COMMIT           HOLD          CASCADE_FALLBACK
                │               │                  │
        speculative audio  chờ anchor      ASR→LLM→TTS (= Giai đoạn 1)
                └───────────────┴──────────────────┘
                                │
                    Output commit controller
                (100-250ms buffer, clause-boundary handoff)
                                │
                             Speaker
```

### 3 chế độ vận hành theo rủi ro nội dung

| Mode | Khi nào dùng | Fast path được phép nói gì |
| --- | --- | --- |
| A — S2S speculative | Small talk, brainstorm, nội dung ít rủi ro | Toàn bộ phản hồi, verifier kiểm tra nhẹ |
| B — Anchored S2S | Phần lớn agent nghiệp vụ | Chỉ ack ("Để mình kiểm tra") — nội dung factual đợi anchor |
| C — Cascade fallback | Thanh toán/y tế/pháp lý/xóa dữ liệu/model confidence thấp/anchor-fastpath bất đồng | Không nói gì — toàn bộ qua cascaded pipeline hiện tại |

### Track A0-A6 (đổi số từ "Giai đoạn 0-6" trong đề xuất gốc để không trùng với Giai
đoạn 0-3 đã dùng ở mục 2)

| Track | Việc chính | Cần GPU? | Go/no-go | Trạng thái |
| --- | --- | --- | --- | --- |
| A0 | Bộ benchmark 300-500 câu tiếng Việt (3 miền, ngắt lời, số/ngày/tên, tình huống nhạy cảm) + đo latency/WER/interruption | Không | ASR stable-prefix đủ tốt, TTS đọc đúng số/ngày/tên, anchor TTFT kịp handoff | **Việc tiếp theo, chưa bắt đầu** |
| A1 | Audio fan-out — chạy song song fast path + anchor path, CHƯA nối output, chỉ log | Có (fast path) | prefix_accept_rate đủ cao để hybrid có giá trị | Chưa bắt đầu |
| A2 | Verifier (rule + entity + NLI) theo risk policy | Không (verifier nhẹ) | — | Chưa bắt đầu |
| A3 | Relay handoff (committed prefix, clause-boundary, commit buffer, cancellation) | Có | Text handoff ổn định trước khi làm audio handoff | Chưa bắt đầu |
| A4 | Tool/RAG qua action channel, phân loại read-only/reversible/irreversible/sensitive | Không thêm | — | Chưa bắt đầu, có thể tái dùng Event Bus/Tool Service hiện có |
| A5 | Fine-tune fast path (Moshi) nói tiếng Việt — user/assistant/inner-monologue stream | Có, nhiều tuần | — | Chưa bắt đầu, rủi ro cao nhất, phụ thuộc thuê GPU cloud (giống Track D cũ) |
| A6 | Production hardening (tracing, session recovery, circuit breaker sang cascade...) | — | — | Chưa bắt đầu |

### Ràng buộc phần cứng đo thật tại thời điểm quyết định (2026-07-17)

```
GPU: NVIDIA GB10 — utilization 96% (tiến trình khác của user tts02, không liên quan)
RAM: 9.4GB free / 121GB (46GB available tính cả cache)
```

Máy dùng chung, đúng rủi ro đã ghi ở mục "Rủi ro đã biết"
(`docs/platform-architecture.md`) — hiện KHÔNG có chỗ chạy Moshi hay bất kỳ fast-path
model GPU nào để bắt đầu prototyping. Đây là lý do Track A0 (thuần CPU, không đụng GPU)
là điểm bắt đầu đúng bất kể GPU đang bận hay không — không phải vì né việc, mà vì A0
không phụ thuộc tài nguyên đang thiếu.

### Model ứng viên theo path (CHƯA VERIFY — cần tự kiểm chứng license/tồn tại/chất
lượng thật trước khi chọn, xem lưu ý nguồn ở đầu mục)

| Path | Ứng viên | Ghi chú cần tự kiểm chứng |
| --- | --- | --- |
| Fast path (S2S) | Moshi/Mimi | Claim ~160-200ms latency là số công bố của tác giả, chưa tự đo; license và checkpoint tiếng Việt chưa xác nhận |
| Anchor ASR | Qwen3-ASR-0.6B/1.7B | Chưa xác nhận tồn tại/license — nếu không có, `selfhost/asr.py` (Zipformer, đã verify WER 2.7%) đã sẵn dùng được |
| Anchor LLM | Qwen3-8B (đã dùng) / Qwen3-30B-A3B-Instruct-2507 | Qwen3-8B-AWQ đã verify chạy thật qua vLLM — 30B-A3B chưa thử, cần VRAM lớn hơn |
| TTS renderer chung | VieNeu-TTS (đã verify) / Kani TTS Vie / Valtec TTS | Kani/Valtec chưa xác nhận license, độ ổn định, pronunciation — VieNeu-TTS là lựa chọn an toàn đã có |
| Verifier | Rule + entity comparator + NLI tiếng Việt | Model NLI tiếng Việt chất lượng đủ dùng chưa xác nhận có sẵn — có thể cần tự train/chọn nhỏ |

### KPI mục tiêu

| Metric | MVP | Production |
| --- | --- | --- |
| First acknowledgement P50 | < 400ms | < 250ms |
| First factual audio P50 | < 900ms | < 600ms |
| Interruption stop P50 | < 300ms | < 200ms |
| Prefix contradiction | < 5% | < 1% |
| Sensitive factual error | 0 trong test gate | 0 |
| Cascade fallback rate | < 50% | < 20-30% |

Metric quan trọng nhất không phải first-audio latency đơn thuần mà là **first verified
useful audio latency** — phát nhanh rồi phải sửa lại nội dung không tính là latency tốt.

### Trạng thái

Chỉ là quyết định kiến trúc, ghi lại ngày 2026-07-17 — **chưa có dòng code nào** cho các
track A0-A6. Việc tiếp theo đã chọn cùng người dùng: Track A0 (bộ benchmark + harness),
tái dùng `eval/testset.py`/`eval/asr_wer.py`/`eval/duplex_bench` đã có sẵn.
