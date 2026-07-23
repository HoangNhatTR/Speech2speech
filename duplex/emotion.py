"""Giai đoạn 2 — Kênh cảm xúc song song (docs/roadmap.md mục 6): "SER trích nhãn cảm
xúc... → chèn vào context LLM dạng thẻ → LLM quyết định giọng đáp".

QUAN TRỌNG — đây KHÔNG phải SER thật: model thật roadmap đề xuất (emotion2vec+ đóng
băng + classifier nhẹ, hoặc SenseVoice) phân tích PROSODY trên AUDIO (tốc độ nói, cao
độ, ngắt quãng...), chưa được tích hợp ở đây (cần model + có thể cần GPU để chạy đủ
nhanh, ngoài phạm vi lần sửa này). File này chỉ dựng ĐƯỜNG DÂY end-to-end
"phát hiện -> chèn tag context -> LLM đọc & phản hồi" chạy được ngay hôm nay bằng một
heuristic từ khoá trên TEXT (transcript), để không phải chờ có model SER mới biết luồng
còn lại (context tag, system prompt) có hoạt động không. Khi có model SER thật, chỉ cần
thay `_classify_heuristic` bằng lệnh gọi model — phần còn lại (inject vào context) giữ
nguyên.

EMOTION_BACKEND=none (mặc định) — tắt hẳn, không đổi hành vi hiện tại của bot.py.
EMOTION_BACKEND=heuristic — bật đường dây thử nghiệm mô tả ở trên.

CHƯA VERIFY bằng hội thoại thật. Xem duplex/interruption_recovery.py về cách
`LLMContext.add_message(role="system", ...)` được Anthropic adapter xử lý (chuyển
thành "user" ở message tiếp theo sau system prompt đầu) — cùng cơ chế áp dụng ở đây.
"""

import re

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from gateway import runtime_config

# Khởi điểm rất thô — thay bằng model SER thật (roadmap.md mục 6) khi có điều kiện.
# Chỉ bắt các dấu hiệu RÕ RÀNG trong lời nói, tránh false positive trên câu trung tính.
_FRUSTRATED_PATTERNS = [
    r"\bbực\b",
    r"\bkhó chịu\b",
    r"\btức\b",
    r"\bchán\b",
    r"\bmệt mỏi\b",
    r"\blâu (vậy|thế|quá)\b",
    r"\bsai (rồi|hoài)\b",
    r"\bkhông hiểu (gì|nổi)\b",
]
_HAPPY_PATTERNS = [
    r"\bcảm ơn\b",
    r"\btuyệt\b",
    r"\brất tốt\b",
    r"\bhài lòng\b",
]

_FRUSTRATED_RE = re.compile("|".join(_FRUSTRATED_PATTERNS), re.IGNORECASE)
_HAPPY_RE = re.compile("|".join(_HAPPY_PATTERNS), re.IGNORECASE)


def _classify_heuristic(text: str) -> str | None:
    """Trả về nhãn cảm xúc nếu bắt được dấu hiệu rõ ràng, None nếu trung tính/không rõ.
    KHÔNG phải SER thật — chỉ khớp từ khoá trên transcript, xem docstring đầu file."""
    if _FRUSTRATED_RE.search(text):
        return "frustrated"
    if _HAPPY_RE.search(text):
        return "happy"
    return None


class EmotionTaggingProcessor(FrameProcessor):
    """Đặt trước `context_aggregator.user()`. Khi phát hiện cảm xúc rõ ràng trong lượt
    nói vừa hoàn tất của người dùng, chèn một message hệ thống dạng
    `[user_emotion: <nhãn>]` vào context trước khi LLM xử lý lượt đó — đúng format
    roadmap.md mục 6 đề xuất."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._backend = runtime_config.get("EMOTION_BACKEND", "none")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._backend == "heuristic" and isinstance(frame, TranscriptionFrame):
            emotion = _classify_heuristic(frame.text)
            if emotion:
                self._context.add_message(
                    {"role": "system", "content": f"[user_emotion: {emotion}]"}
                )
                logger.debug(f"[duplex] Cảm xúc (heuristic, chưa phải SER thật): {emotion}")

        await self.push_frame(frame, direction)
