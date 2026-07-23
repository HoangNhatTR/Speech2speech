"""Kịch bản chấm điểm cho Vietnamese Turn-Taking Classifier Bench (xem __init__.py về
phạm vi). Mỗi Scenario mô phỏng một cửa sổ "bot đang nói" (hoặc bot im lặng, cho nhóm
normal_turn) trong đó người dùng phát ra một chuỗi partial-transcript (giống ASR
streaming thật: text dài dần theo thời gian), kèm nhãn đúng (expected_turn_started) do
người viết kịch bản này gán thủ công dựa trên trực giác ngôn ngữ tiếng Việt — KHÔNG lấy
từ một bộ dữ liệu hội thoại thật đã gán nhãn độc lập, nên coi là smoke-test/seed set, có
thể mở rộng thêm khi có dữ liệu hội thoại thật (xem docs/roadmap.md mục 8).

t_ms: mốc thời gian giả lập (mô phỏng ASR partial cập nhật dần mỗi ~100-150ms, khớp con
số trong docs/roadmap.md mục 4) — dùng để tính "detection latency" mô phỏng, KHÔNG phải
độ trễ đo trên hệ thống thật (không có audio/API thật chạy qua).
"""

from dataclasses import dataclass, field
from typing import Literal

FrameKind = Literal["interim", "final"]
Category = Literal["backchannel", "real_interrupt", "normal_turn"]


@dataclass
class Step:
    frame: FrameKind
    text: str
    t_ms: float


@dataclass
class Scenario:
    id: str
    category: Category
    bot_speaking_before: bool
    steps: list[Step]
    expected_turn_started: bool
    note: str = ""


def _typed(text: str, t0: float = 120.0, step_ms: float = 110.0) -> list[Step]:
    """Mô phỏng ASR gõ dần từng từ một thành các InterimTranscriptionFrame, kết thúc
    bằng một TranscriptionFrame (final) chứa câu đầy đủ — giống hành vi streaming STT
    thật (Deepgram/Zipformer đều trả interim trước, final sau)."""
    words = text.split()
    steps = []
    for i in range(1, len(words) + 1):
        steps.append(Step("interim", " ".join(words[:i]), t0 + (i - 1) * step_ms))
    steps.append(Step("final", text, t0 + len(words) * step_ms))
    return steps


SCENARIOS: list[Scenario] = [
    # --- Backchannel: bot đang nói, user chêm từ đệm, KHÔNG được coi là ngắt lời ---
    Scenario("bc_da", "backchannel", True, _typed("Dạ"), False,
             "Từ đệm đơn, phổ biến nhất khi nghe đồng ý."),
    Scenario("bc_vang_a", "backchannel", True, _typed("Vâng ạ"), False,
             "Từ đệm 2 từ, lịch sự."),
    Scenario("bc_um", "backchannel", True, _typed("Ừm"), False, "Từ đệm ngập ngừng."),
    Scenario("bc_da_vang", "backchannel", True, _typed("Dạ vâng"), False,
             "Tổ hợp 2 từ đệm liền nhau."),
    Scenario("bc_o_o", "backchannel", True, _typed("Ờ ờ"), False, "Lặp từ đệm."),
    Scenario("bc_uhm", "backchannel", True, _typed("Uhm"), False, "Biến thể chính tả của ừm."),
    Scenario("bc_vang_vang_a", "backchannel", True, _typed("Vâng vâng ạ"), False,
             "3 từ đệm, đúng ranh giới max_backchannel_words mặc định."),
    Scenario("bc_a", "backchannel", True, _typed("À"), False, "Từ đệm ngắn, dễ nhầm với ngạc nhiên."),

    # --- Ngắt lời thật: bot đang nói, user thực sự có nội dung mới, PHẢI coi là ngắt ---
    Scenario("int_doi_da", "real_interrupt", True, _typed("Không đợi đã"), True,
             "Ngắt lời rõ ràng, không chứa từ đệm nào."),
    Scenario("int_da_nhung", "real_interrupt", True, _typed("Dạ nhưng mà em muốn hỏi thêm"), True,
             "Bắt đầu bằng từ đệm 'dạ' nhưng câu dài có nội dung thật — bẫy dễ nhầm "
             "thành backchannel nếu chỉ nhìn từ đầu câu."),
    Scenario("int_anh_oi", "real_interrupt", True, _typed("Anh ơi cho em hỏi cái này"), True,
             "Câu hỏi thật chen ngang."),
    Scenario("int_vang_nhung", "real_interrupt", True,
             _typed("Vâng nhưng em cần đổi lịch hẹn"), True,
             "Bắt đầu bằng 'vâng' (thường là từ đệm) nhưng mang yêu cầu thật — ca khó "
             "nhất: không phải mọi câu bắt đầu bằng từ đệm đều là backchannel."),
    Scenario("int_um_nhung", "real_interrupt", True, _typed("Ừm nhưng mà tôi nghĩ khác"), True,
             "Bắt đầu bằng 'ừm' nhưng có ý kiến phản bác thật."),
    Scenario("int_doi_chut", "real_interrupt", True, _typed("Đợi chút"), True,
             "Câu ngắn (2 từ, dưới ngưỡng max_backchannel_words) nhưng không phải từ đệm."),

    # --- Turn bình thường: bot đang im lặng, mọi phát ngôn đều phải được coi là turn mới,
    # kể cả khi nội dung trùng với "từ đệm" (vd trả lời "dạ" cho câu hỏi có/không) ---
    Scenario("norm_da_answer", "normal_turn", False, _typed("Dạ"), True,
             "Bot im lặng, user trả lời 'dạ' cho câu hỏi có/không trước đó — đây là câu "
             "trả lời thật, không phải backchannel, dù trùng từ với nhóm backchannel."),
    Scenario("norm_long", "normal_turn", False,
             _typed("Tôi muốn kiểm tra số dư tài khoản của mình"), True,
             "Turn bình thường, câu dài, bot không đang nói."),
    Scenario("norm_vang", "normal_turn", False, _typed("Vâng"), True,
             "Bot im lặng, 'vâng' đơn là câu trả lời đầy đủ, phải mở turn mới."),
]
