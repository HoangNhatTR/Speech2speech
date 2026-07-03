# Speech2Speech (tiếng Việt)

Trợ lý giọng nói speech-to-speech tiếng Việt, xây theo lộ trình 4 giai đoạn trong
[`speech2speech.pdf`](../speech2speech.pdf) (bản đầy đủ, dễ đọc hơn ở
[`docs/roadmap.md`](docs/roadmap.md)).

## Đang ở giai đoạn nào

**Giai đoạn 0 — Khung xương.** Pipeline cascaded chạy hoàn toàn bằng API thương mại
(chưa có model tiếng Việt tự host):

```
WebRTC mic → Deepgram STT (nova-3, vi) → Claude (Anthropic) → ElevenLabs TTS (eleven_flash_v2_5) → WebRTC loa
```

Mục tiêu giai đoạn này **không phải chất lượng cuối**, mà là có hệ thống streaming chạy
được để chốt UX (barge-in, turn-taking — Pipecat có sẵn), đo latency từng chặng, và làm
eval harness trước khi thay từng khối bằng model tự host ở giai đoạn 1.

> Máy dev hiện tại (GPU 2GB) không đủ VRAM để tự host ASR/LLM/TTS tiếng Việt của giai
> đoạn 1 trở đi (cần ít nhất 1×RTX 4090/A10 + 1×A100/H100 theo lộ trình, mục 10 —
> Phần cứng). Giai đoạn 0 chạy tốt trên máy này vì mọi model đều gọi qua API cloud.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# rồi điền ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID vào .env
```

## Chạy

```powershell
.venv\Scripts\python server.py
```

Mở `http://localhost:7860/client`, cấp quyền micro, và nói chuyện.

Vì sao `server.py` chứ không phải `python bot.py -t webrtc` (cách chuẩn của Pipecat)?
Pipecat-ai bản mới nhất còn hỗ trợ Python 3.10 (0.0.108) có một bug tương thích:
`pipecat.runner.run` import `http.HTTPMethod`, chỉ có từ Python 3.11. `server.py` dựng
lại đúng phần cần thiết (route WebRTC offer/answer + giao diện thử) mà không đụng module
đó. Nếu sau này nâng lên Python 3.11+, có thể quay lại dùng `python bot.py -t webrtc`
thẳng và xoá `server.py`.

## Đọc log để đánh giá (eval harness có sẵn từ Pipecat, không cần code thêm)

Console sẽ in:
- `[EVAL] TTFA (kết nối → tiếng nói đầu tiên)` — mục tiêu < 1s theo lộ trình.
- `[EVAL] Turn latency (user ngừng nói → bot bắt đầu nói)` — độ trễ mỗi lượt hội thoại.
- Transcript STT theo thời gian thực, và TTFB (time-to-first-byte) của từng service
  (Deepgram/Claude/ElevenLabs) qua `MetricsLogObserver`.

Đây chính là "eval harness" mà giai đoạn 0 yêu cầu (đo latency từng chặng trước khi thay
model tự host).

## Lưu ý chất lượng ở giai đoạn này

- ASR tiếng Việt qua Deepgram sẽ không tốt bằng Zipformer tự host của giai đoạn 1 —
  chấp nhận được, vì mục tiêu ở đây là đo UX/latency, không phải WER.
- ElevenLabs chỉ có `eleven_flash_v2_5` hỗ trợ tiếng Việt (không phải
  `eleven_multilingual_v2`) — đã set cứng trong `bot.py`.
- Barge-in/interruption dùng cơ chế mặc định của Pipecat (Silero VAD). Barge-in "đúng
  chuẩn" theo lộ trình (phân loại backchannel "dạ/ừm/vâng", AEC tường minh, phục hồi
  context khi bị ngắt) là việc của giai đoạn 1.

## Bước tiếp theo (giai đoạn 1)

Thay từng khối bằng model tự host cho tiếng Việt: ASR → Zipformer streaming
(sherpa-onnx), LLM → Qwen3 qua vLLM, TTS → F5-TTS-Vietnamese/viXTTS. Cần thuê GPU cloud
(A100/H100) vì máy dev hiện tại không đủ VRAM. Xem mục 2–3 trong
[`docs/roadmap.md`](docs/roadmap.md).
