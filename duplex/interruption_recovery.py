"""Giai đoạn 2 — Phục hồi context khi bị ngắt lời (docs/roadmap.md mục 4, tầng 4):
"Khi cắt TTS phải biết bot đã phát đến từ nào để ghi vào lịch sử hội thoại dạng
'assistant: ... (bị ngắt)', nếu không LLM sẽ tưởng người dùng đã nghe hết."

Đặt processor này SAU `tts`, TRƯỚC `transport.output()` trong pipeline: mọi
`TTSTextFrame` đi qua đây là phần bot ĐÃ thực sự được tổng hợp thành âm thanh (không
phải toàn bộ câu LLM sinh ra, vì phần chưa kịp tới TTS thì chưa tính là "đã nói"). Khi
gặp `InterruptionFrame` trong lúc đang có phần chưa "chốt", ghi một message vào
`context`. Không có word timestamp nên phần này chỉ là text đã tổng hợp/xếp hàng, không
được coi là vị trí phát chính xác tới từng từ. Buffer phải xóa khi bot nói xong, nếu
không lần ngắt ở lượt sau sẽ lẫn nội dung của mọi lượt trước.

Đã xác nhận bằng cách đọc source Pipecat 0.0.108 thật:
- `TTSTextFrame`/`InterruptionFrame` tồn tại đúng như dùng ở đây (không đoán tên).
- `LLMContext.add_message({"role": "system", ...})` là API công khai thật.
- Với backend Anthropic (`services/llm_client.py`, `AnthropicLLMService`): đọc
  `pipecat/adapters/services/anthropic_adapter.py` xác nhận adapter tự động chuyển mọi
  message "system" xuất hiện SAU system prompt đầu tiên thành "user" (Anthropic API
  không nhận role "system" giữa hội thoại), và tự gộp các message liên tiếp cùng role
  làm một — nghĩa là message ta thêm ở đây sẽ tới LLM dưới dạng nội dung "user" (có thể
  gộp chung với lượt nói thật ngay sau), không phải lỗi, chỉ cần biết để không ngạc
  nhiên khi debug log.

CHƯA VERIFY bằng hội thoại thật (cần ANTHROPIC_API_KEY + tự ngắt lời bot để nghe/xem log
thật) — đặc biệt là thứ tự tương đối giữa message này và message assistant do
`context_aggregator.assistant()` tự ghi (nằm sau processor này trong pipeline, có thể
chạy lệch nhịp một chút). Sai lệch thứ tự chỉ ảnh hưởng độ "mượt" của ngữ cảnh, không
làm hỏng hội thoại — không phải chỗ nên tin tưởng tuyệt đối nếu chưa nghe thử.
"""

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class InterruptionRecoveryProcessor(FrameProcessor):
    """Gom text đã thực sự phát ra loa; khi bị ngắt giữa chừng, ghi lại vào `context`."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._spoken_buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._track_frame(frame)
        await self.push_frame(frame, direction)

    def _track_frame(self, frame: Frame) -> None:
        """Cập nhật buffer/context; tách riêng để test không cần dựng cả PipelineTask."""
        if isinstance(frame, TTSTextFrame):
            self._spoken_buffer.append(frame.text)
        elif isinstance(frame, InterruptionFrame) and self._spoken_buffer:
            spoken = " ".join(self._spoken_buffer).strip()
            self._spoken_buffer.clear()
            note = (
                "[Hệ thống: lượt trả lời TRƯỚC của bạn bị người dùng ngắt lời giữa "
                f'chừng. Phần đã được tổng hợp/xếp phát trước lúc ngắt: "{spoken}". '
                "Không có word timestamp nên người dùng có thể chỉ nghe một phần; đừng "
                "giả định họ đã nghe trọn nội dung này.]"
            )
            self._context.add_message({"role": "system", "content": note})
            logger.debug(f"[duplex] Ngắt lời giữa chừng, đã nói tới: {spoken!r}")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._spoken_buffer.clear()
