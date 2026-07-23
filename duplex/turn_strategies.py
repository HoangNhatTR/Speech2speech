"""Giai đoạn 2 — Phân loại ngắt lời (docs/roadmap.md mục 4, tầng 3): "200ms tiếng nói
khi bot đang phát chưa chắc là barge-in — có thể là backchannel... đáng fine-tune cho
tiếng Việt vì backchannel 'dạ', 'ừm', 'vâng' rất đặc trưng."

Xác nhận bằng cách đọc source `pipecat` 0.0.108 thật (không đoán API):
- `pipecat.turns.user_start.base_user_turn_start_strategy.BaseUserTurnStartStrategy` là
  interface chuẩn cho việc quyết định khi nào một lượt nói của người dùng "bắt đầu".
- Mặc định (`UserTurnStrategies.__post_init__` khi không truyền gì), pipecat đã dùng
  `[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]` làm start-strategy
  và `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())` làm stop-strategy —
  nghĩa là bot.py hiện tại ĐÃ dùng smart-turn v3 (bundled ONNX, chạy CPU, không cần tải
  thêm gì) cho việc phát hiện người dùng nói XONG, hoàn toàn miễn phí, không cần viết gì
  thêm. Đây là phát hiện khi rà lại pipecat cho việc này — "semantic endpointing" mà
  roadmap.md mục 2 (Giai đoạn 2) nhắc tới coi như đã có sẵn ở nửa "stop" từ Giai đoạn 0.
- Việc còn thiếu, và là lý do file này tồn tại, là nửa "start" khi bot ĐANG nói: mặc định
  `VADUserTurnStartStrategy` ngắt ngay khi có bất kỳ âm thanh nào — không phân biệt được
  "dạ" (backchannel) với một câu ngắt lời thật.
  `pipecat.turns.user_start.min_words_user_turn_start_strategy.MinWordsUserTurnStartStrategy`
  đã có sẵn cơ chế đúng hướng (số từ tối thiểu khi bot đang nói) nhưng chỉ đếm SỐ TỪ, không
  biết TỪ GÌ — "dạ vâng ạ" (3 từ) vẫn sẽ bị coi là ngắt lời thật với min_words=2.
  Class dưới đây thay bằng danh sách từ backchannel tiếng Việt cụ thể.

Để backend local vẫn barge-in nhanh dù Zipformer không có interim transcript, strategy
còn nghe VAD: khi bot đang nói, nó chờ một cửa sổ ngắn (mặc định 200ms) cho ASR nhận ra
backchannel; hết cửa sổ thì ngắt bot ngay, không chờ người dùng nói xong. Khi bot im
lặng, VAD mở lượt tức thời như strategy mặc định.
"""

import asyncio
import re
from typing import Iterable, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

# Khởi điểm, không đầy đủ — nên mở rộng dựa trên dữ liệu hội thoại thật thu thập được
# (đúng tinh thần roadmap.md mục 4: "đáng fine-tune cho tiếng Việt"). Chỉ chứa các TỪ
# ĐƠN không mơ hồ (loại "được"/"rồi" ra vì chúng cũng là câu trả lời thật hợp lệ). Khớp
# theo kiểu "mọi từ trong câu đều là từ đệm" (bag-of-words) để tự nhiên bao được các tổ
# hợp như "dạ vâng ạ", "vâng vâng ạ"... mà không cần liệt kê hết mọi hoán vị.
DEFAULT_BACKCHANNEL_WORDS = {
    "dạ",
    "vâng",
    "ạ",
    "ừ",
    "ừm",
    "ờ",
    "à",
    "uh",
    "uhm",
}

_PUNCT_RE = re.compile(r"[.,!?;:…]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip().lower()


class VietnameseBackchannelTurnStartStrategy(BaseUserTurnStartStrategy):
    """Start-strategy thay cho `VADUserTurnStartStrategy` mặc định: khi bot đang nói,
    nếu transcript (interim hoặc final) khớp một backchannel tiếng Việt đã biết thì
    KHÔNG coi là bắt đầu lượt mới (bot tiếp tục nói) — ngược lại (câu dài hơn, hoặc
    không khớp danh sách, hoặc bot không đang nói) thì coi là bắt đầu lượt như bình
    thường.

    Khi bot đang nói, VAD khởi động timer `barge_in_delay_ms`: transcript backchannel
    tới trong cửa sổ này sẽ huỷ timer; transcript khác sẽ ngắt ngay; nếu ASR không có
    interim (nhánh local) thì timer tự ngắt bot khi hết hạn. Đây là điểm cân bằng giữa
    barge-in tức thời và false interruption do "dạ/vâng/ừm".
    """

    def __init__(
        self,
        *,
        backchannel_words: Optional[Iterable[str]] = None,
        max_backchannel_words: int = 3,
        use_interim: bool = True,
        barge_in_delay_ms: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._backchannel_words = {
            _normalize(w) for w in (backchannel_words or DEFAULT_BACKCHANNEL_WORDS)
        }
        self._max_backchannel_words = max_backchannel_words
        self._use_interim = use_interim
        self._barge_in_delay_s = max(0, barge_in_delay_ms) / 1000
        self._bot_speaking = False
        self._pending_vad_task: asyncio.Task | None = None

    def _cancel_pending_vad(self) -> None:
        task = self._pending_vad_task
        self._pending_vad_task = None
        if task and not task.done():
            task.cancel()

    async def _trigger_after_vad_grace(self) -> None:
        try:
            await asyncio.sleep(self._barge_in_delay_s)
        except asyncio.CancelledError:
            return

        # Xoá reference trước khi trigger: controller sẽ gọi reset() đồng bộ khi turn
        # bắt đầu, tránh reset tự cancel chính task đang chạy này.
        self._pending_vad_task = None
        if self._bot_speaking:
            logger.debug(
                "[duplex] VAD barge-in sau {:.0f}ms "
                "(không có backchannel transcript kịp thời)",
                self._barge_in_delay_s * 1000,
            )
            await self.trigger_user_turn_started()

    async def reset(self):
        await super().reset()
        self._cancel_pending_vad()
        self._bot_speaking = False

    async def cleanup(self):
        self._cancel_pending_vad()
        await super().cleanup()

    def _looks_like_backchannel(self, text: str) -> bool:
        normalized = _normalize(text)
        if not normalized:
            return False
        words = normalized.split()
        if len(words) > self._max_backchannel_words:
            return False
        return all(w in self._backchannel_words for w in words)

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._pending_vad_task:
                self._cancel_pending_vad()
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            if not self._pending_vad_task:
                self._pending_vad_task = asyncio.create_task(
                    self._trigger_after_vad_grace(),
                    name="vietnamese-backchannel-vad-grace",
                )
        elif isinstance(frame, TranscriptionFrame):
            return await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            return await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        if self._bot_speaking and self._looks_like_backchannel(frame.text):
            self._cancel_pending_vad()
            logger.debug(f"[duplex] Backchannel bỏ qua, bot tiếp tục nói: {frame.text!r}")
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE

        self._cancel_pending_vad()
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP
