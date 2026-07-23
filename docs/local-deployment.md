# Triển khai Speech2Speech real-time trên local

Tài liệu này chỉ áp dụng cho repo `Speech2speech`. Mục tiêu là dựng một stack có thể
tự kiểm tra trên một máy, theo hai profile `small` và `medium`, đồng thời giữ đường nâng
cấp sang mô hình speech-to-speech audio-token.

## 1. Trạng thái kiến trúc hiện tại

Đường trả lời đang hoạt động và an toàn vẫn là cascaded Anchor path:

```text
Micro WebRTC
  -> AudioHub (fan-out audio, ring buffer hữu hạn)
  -> VAD + Zipformer ASR
  -> Qwen3-8B-AWQ qua vLLM
  -> VieNeu-TTS streaming
  -> WebRTC loa

AudioHub
  -> audio-token backend probe (S2S_MODE=shadow)
  -> SegmentPolicy + SessionOrchestrator
  -> chỉ đo/ghi nhận, chưa phát audio ra loa
```

Đây **chưa phải** mô hình end-to-end audio-token hoàn chỉnh: backend `probe` xác minh
media/control plane nhưng không giả làm Moshi/Mimi. Thiết kế dual-path cho phép thêm
Moshi sau này mà không bỏ ASR/LLM/tool/RAG và không cho câu chưa được xác minh phát ra
loa.

## 2. Hai profile local

| Giới hạn | `small` | `medium` |
|---|---:|---:|
| Voice session tối đa | 2 | 8 |
| Request đồng thời mỗi session | 1 | 2 |
| Request đồng thời toàn gateway | 4 | 16 |
| vLLM max model length | 4.096 | 8.192 |
| vLLM max sequences | 4 | 16 |
| Audio ring buffer | 10 giây | 30 giây |
| Runtime dispatcher | asyncio, 1 process | asyncio, 1 process |
| S2S ban đầu | `shadow` | `shadow` |

Các giá trị trên là mặc định trong `config/profiles/*.env`. `.env` ưu tiên hơn profile,
và biến export ở shell ưu tiên cao nhất. Nếu một vLLM/TTS/NATS đã chạy và khỏe, launcher
sẽ dùng lại service đó; các tham số profile không restart hay thay đổi service ngoài
quyền quản lý của launcher.

Chọn `small` để phát triển một người dùng và đo pipeline. Chỉ chuyển `medium` sau khi
đã đo VRAM/RAM, TTFA và realtime factor với tải đồng thời thực tế.

## 3. Chuẩn bị lần đầu

Trong `.env`, cấu hình local tối thiểu:

```dotenv
STT_BACKEND=local
LLM_BACKEND=local
TTS_BACKEND=local
ASR_LOCAL_ENGINE=zipformer
VLLM_BASE_URL=http://localhost:8000/v1
VIENEU_SERVER_URL=http://localhost:8100
S2S_MODE=shadow
S2S_SHADOW_BACKEND=probe
GATEWAY_AUTH_DISABLED=true
```

Các môi trường `.venv`, `.venv-vllm`, `.venv-tts`, model Zipformer và NATS phải được
cài theo README. Kiểm tra không làm thay đổi hệ thống:

```bash
cd /home/ai01/AIHoang/Speech2speech
.venv/bin/python -m scripts.local_stack doctor --profile small
```

Mọi dòng phải là `OK`. Có thể thêm `--skip-frontend` nếu chỉ test API/WebRTC gateway.

## 4. Start, kiểm tra và stop

Khởi động profile nhỏ:

```bash
.venv/bin/python -m scripts.local_stack start --profile small
.venv/bin/python -m scripts.local_stack status
```

Các địa chỉ:

- Voice WebRTC: `http://localhost:7860/client/`
- Readiness JSON: `http://localhost:7860/api/ready`
- Dashboard/Settings: `http://localhost:5173`

Smoke nhanh chỉ kiểm tra wiring:

```bash
.venv/bin/python -m scripts.local_smoke --quick
```

Smoke đầy đủ gọi thật local Qwen và VieNeu, đồng thời xác nhận có audio chunk đầu:

```bash
.venv/bin/python -m scripts.local_smoke
```

Kết quả hợp lệ phải kết thúc bằng `PASS: local stack smoke test`. Dừng stack:

```bash
.venv/bin/python -m scripts.local_stack stop
```

Lệnh `stop` chỉ gửi tín hiệu tới PID đã ghi trong `.runtime/local_stack.json`; NATS,
vLLM, TTS hoặc frontend đã chạy từ trước và được đánh dấu `external` sẽ được giữ lại.

## 5. Test hội thoại bằng microphone

1. Chạy full smoke trước để tách lỗi model khỏi lỗi trình duyệt.
2. Mở `http://localhost:7860/client/` trên chính máy chạy stack.
3. Cấp quyền microphone, bấm Connect và chờ trạng thái connected.
4. Nói một câu ngắn: “Xin chào, hãy giới thiệu ngắn gọn.”
5. Khi bot đang nói, nói “dừng lại” để kiểm tra barge-in.
6. Nói “dạ”, “ừm” ngắn để kiểm tra backchannel không cắt bot sai.
7. Theo dõi log gateway để xem ASR transcript, TTFA, turn latency và interruption.

Không mở gateway HTTP ra một máy khác để thu micro: trình duyệt thường chỉ cấp quyền
microphone cho secure context (`https`) hoặc `localhost`. Khi test từ máy khác cần đặt
reverse proxy HTTPS và bật auth; không đổi `HOST` thành `0.0.0.0` rồi để auth tắt.

## 6. Log và xử lý lỗi

Launcher ghi log dưới `logs/`:

- `logs/local_gateway.log`: WebRTC, ASR, LLM, TTS, barge-in và latency.
- `logs/local_dispatcher.log`: NATS subjects và lỗi service.
- `logs/local_vllm.log`, `logs/local_tts.log`: có khi launcher tự tạo hai backend.
- `logs/local_frontend.log`: frontend Vite khi được launcher tạo.

Các kiểm tra nhanh:

```bash
curl -s http://localhost:7860/api/ready
tail -f logs/local_gateway.log
.venv/bin/python -m scripts.local_stack status
```

Nếu `ready` trả 503, trường `checks` chỉ rõ khối hỏng. Nếu launcher báo đã có process
managed, dùng `status`; chỉ dùng `stop` trước khi start lại, không xóa PID manifest thủ
công. Nếu GPU thiếu bộ nhớ, giảm tải khác, giữ profile `small`, giảm
`VLLM_GPU_MEMORY_UTILIZATION` hoặc chuyển VieNeu về CPU để chẩn đoán.

## 7. Thứ tự nâng lên audio-token S2S

Không chuyển thẳng `S2S_MODE=primary`. Thực hiện theo gate:

1. **Baseline:** giữ Anchor path local đạt smoke và thu latency/hội thoại micro.
2. **Codec gate:** chạy `eval.mimi_codec` trên tập audio tiếng Việt người thật; đo SI-SDR,
   STOI/PESQ hoặc MOS và lỗi tên riêng/số sau resynthesis.
3. **Moshi adapter:** `moshi_ws` đã triển khai đúng protocol chính thức; model chạy
   process/venv riêng, nhận frame từ `AudioHub`; gateway không import torch/checkpoint.
4. **Shadow:** Moshi chạy song song nhưng tuyệt đối không phát ra loa; đo first audio,
   semantic agreement với Anchor và GPU/RAM.
5. **Ack-only:** chỉ cho SegmentPolicy phát các acknowledgement an toàn, không chứa
   số liệu, tên riêng, kết quả tool hay claim thực tế.
6. **Speculative:** cho phép prefix có điều kiện; khi ASR/LLM/tool anchor mâu thuẫn thì
   tăng `revision_id`, hủy audio cũ và fallback sang VieNeu.
7. **Primary có điều kiện:** chỉ sau khi benchmark tiếng Việt, barge-in, tool/RAG,
   hallucination và soak test đều đạt ngưỡng; luôn giữ cascade fallback.

Chi tiết model/dataset/benchmark và tiêu chí train nằm trong
[`ke-hoach-cong-viec-train-test-s2s.md`](ke-hoach-cong-viec-train-test-s2s.md). Mimi và
Moshi dùng `.venv-s2s` riêng; không tự tải checkpoint khi gateway khởi động.

### Chạy codec gate và Moshi shadow

Tạo môi trường tách biệt, với cache nằm trong repo:

```bash
./scripts/setup_s2s_env.sh
```

Chạy Mimi trước bằng CPU để không tranh GPU với stack đang phục vụ:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv-s2s/bin/python -m eval.mimi_codec \
  --input <audio-vietnamese.wav> \
  --output eval/results/mimi-reconstructed.wav \
  --result-json eval/results/mimi-codec.json --device cpu

.venv/bin/python -m eval.mimi_asr_compare \
  --original <audio-vietnamese.wav> \
  --reconstructed eval/results/mimi-reconstructed.wav \
  --transcript '<transcript chuẩn>' \
  --result-json eval/results/mimi-asr.json
```

Chỉ khi GPU/RAM đủ cho Moshi 7B, dừng launcher hiện tại và bật sidecar thật:

```bash
.venv/bin/python -m scripts.local_stack stop
.venv/bin/python -m scripts.local_stack doctor --profile small --with-moshi
.venv/bin/python -m scripts.local_stack start --profile small --with-moshi
```

Launcher ép `S2S_MODE=shadow`, giữ Hugging Face cache trong `.cache/huggingface`, theo
dõi sidecar ở `logs/local_moshi.log` và dừng sớm nếu process model chết. Server PyTorch
Moshi hiện chỉ phục vụ một inference session đồng thời; session khác timeout fast path
và tiếp tục bằng Anchor. Checkpoint gốc chưa được xem là tiếng Việt production.

## 8. Tiêu chí hoàn tất một vòng local

- `doctor`, `status`, readiness và full smoke đều PASS.
- Kết nối/đóng WebRTC lặp lại không để session treo.
- Barge-in cắt audio đúng; backchannel ngắn không cắt sai.
- Không có hàng đợi audio tăng vô hạn; `AudioHub` drop consumer chậm độc lập.
- p50/p95 TTFA và turn latency được lưu trước/sau mỗi thay đổi model.
- Khi S2S lỗi hoặc timeout, Anchor path vẫn trả lời và tool/RAG không bị bỏ qua.
