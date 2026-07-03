# Lộ trình Speech2Speech tiếng Việt

> Bản markdown dễ đọc/dễ grep của `speech2speech.pdf` (trích xuất bằng `pdftotext`,
> dọn lại bảng biểu). Nếu có sai lệch, file PDF gốc là nguồn chính xác.

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
| ASR streaming | Zipformer-30M-RNNT train trên ~6.000 giờ tiếng Việt, chạy qua sherpa-onnx, SOTA trên VLSP2020/2023 — streaming thật, độ trễ thấp; PhoWhisper (VinAI) cho offline/rescoring | faster-whisper (large-v3-turbo) chunked; Deepgram/Google STT làm fallback |
| VAD + turn | Silero VAD v5 hoặc TEN VAD; smart-turn v3 (Pipecat) / LiveKit turn detector, fine-tune thêm tiếng Việt | Semantic endpointing bằng LLM nhỏ (Qwen3-0.6B) đọc partial transcript |
| LLM anchor | Qwen3 8B–32B (tiếng Việt mạnh, function calling tốt, vLLM streaming); thay thế: Llama 3.3, SeaLLM/Sailor cho Đông Nam Á | Claude/GPT qua API nếu ưu tiên chất lượng tool-use |
| TTS streaming | F5-TTS-Vietnamese train trên viVoice (chất lượng cao, chunk theo câu); viXTTS — XTTS fine-tune trên viVoice, có voice cloning và đa ngôn ngữ (streaming <200ms); VietTTS của dangvansam — cloning qua audio prompt, API tương thích OpenAI; VieNeu-TTS v2 — cloning tức thì, chạy real-time trên CPU, hỗ trợ code-switching Việt–Anh | CosyVoice 2/3 (streaming 2 chiều, emotion instruct); thương mại: ElevenLabs hỗ trợ tiếng Việt kèm cloning (`eleven_flash_v2_5`), CosyVoice v3.5 trên DashScope có voice cloning tiếng Việt |
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
