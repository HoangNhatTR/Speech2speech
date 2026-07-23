"""Tập câu tiếng Việt tham chiếu cho eval harness, có gắn nhãn domain/code-switch để đo
Acc theo từng nhóm trường hợp (quy ước "độ phủ") thay vì chỉ một con số WER gộp — một
model có thể WER tổng ổn nhưng rất tệ ở một domain cụ thể (vd số/tiền tệ, code-switch),
gộp chung sẽ che mất điều đó.

Đây KHÔNG phải benchmark chuẩn (VLSP2020/2023, Common Voice, FLEURS-vi) — những bộ đó
cần đăng ký/tải qua kênh chính thức, chưa tải được trong lần build này. Đây là tập nhỏ tự
viết để eval harness có dữ liệu thật ngay hôm nay: dùng gTTS (Google Translate TTS, miễn
phí, không cần API key) tổng hợp audio từ câu văn bản đã biết trước, rồi đo WER của ASR
trên chính audio đó. Nhược điểm cần biết: audio TTS sạch hơn giọng người thật nhiều
(không nhiễu, không nói lắp, phát âm chuẩn) — WER đo được ở đây là "ideal case", KHÔNG
đại diện cho WER trên giọng nói thật ngoài đời. Coi đây là smoke test hồi quy (so sánh
model A vs B trên cùng điều kiện, và giữa domain/mức nhiễu), không phải con số WER cuối
cùng để công bố.

Giới hạn đã biết, CHƯA khắc phục được bằng harness này (cần dữ liệu thật, xem
docs/roadmap.md mục 8 — Bud500/VLSP2020/VietSuperSpeech):
- gTTS chỉ có 1 giọng tiếng Việt duy nhất — KHÔNG mô phỏng được phương ngữ 3 miền
  (Bắc/Trung/Nam). Bất kỳ tuyên bố "hoạt động tốt trên mọi giọng vùng miền" phải được
  kiểm chứng bằng dữ liệu thật, không phải qua tập này.
- Nhiễu được cộng ở đây (xem eval/asr_wer.py::add_noise) là nhiễu trắng cộng tuyến tính
  sau khi tổng hợp — mô phỏng thô độ suy giảm SNR, không thay được nhiễu môi trường thật
  (tiếng ồn quán cà phê, xe cộ, dội âm phòng...).

Nâng cấp khuyến nghị (chưa làm): thay bằng VIVOS/VLSP2020 test set khi tải được, để có
WER thật trên giọng người, đủ đa dạng vùng miền.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestItem:
    text: str
    domain: str
    code_switch: bool = False
    note: str = ""


TEST_ITEMS: list[TestItem] = [
    # --- Chào hỏi / hội thoại chung ---
    TestItem("Xin chào, hôm nay thời tiết rất đẹp.", "chao_hoi"),
    TestItem("Xin lỗi, tôi không nghe rõ, bạn có thể nhắc lại không?", "chao_hoi"),
    TestItem("Cảm ơn bạn rất nhiều vì đã giúp đỡ tôi.", "chao_hoi"),
    TestItem("Chúc bạn một ngày làm việc hiệu quả và vui vẻ.", "chao_hoi"),
    TestItem("Hà Nội là thủ đô của Việt Nam.", "chao_hoi"),

    # --- Ngân hàng / tài chính (domain nghiệp vụ chính của bot theo datagen script) ---
    TestItem("Tôi muốn kiểm tra số dư tài khoản ngân hàng của mình.", "ngan_hang"),
    TestItem("Anh có thể chuyển tiền giúp tôi được không?", "ngan_hang"),
    TestItem("Cho tôi xem lịch sử giao dịch trong ba tháng gần nhất.", "ngan_hang"),
    TestItem("Tôi cần thanh toán hoá đơn tiền điện tháng này.", "ngan_hang"),
    TestItem("Thẻ tín dụng của tôi bị khoá, phải làm sao để mở lại?", "ngan_hang"),

    # --- Số / ngày giờ / tiền tệ (dễ lẫn khi ASR, cần đo riêng) ---
    TestItem("Bây giờ là mấy giờ rồi nhỉ?", "so_thoi_gian"),
    TestItem("Cuộc hẹn của tôi lúc chín giờ rưỡi sáng ngày mười lăm tháng ba.", "so_thoi_gian"),
    TestItem("Chuyển khoản năm triệu hai trăm nghìn đồng vào tài khoản này.", "so_thoi_gian"),
    TestItem("Số điện thoại của tôi là không chín tám bảy sáu năm bốn ba hai một.", "so_thoi_gian"),

    # --- Code-switching tiếng Việt - Anh (thực tế phổ biến trong hội thoại app/công nghệ) ---
    TestItem("Bạn có thể gửi email confirm lại lịch hẹn cho tôi không?", "code_switch", code_switch=True),
    TestItem("Tôi muốn book vé máy bay đi Sài Gòn vào cuối tuần này.", "code_switch", code_switch=True),
    TestItem("App bị lỗi, tôi không login vào được tài khoản.", "code_switch", code_switch=True),

    # --- Câu dài / phức tạp ---
    TestItem(
        "Món phở bò là một trong những món ăn nổi tiếng của Việt Nam, được nhiều du "
        "khách nước ngoài yêu thích khi đến thăm Hà Nội.",
        "cau_dai",
    ),
    TestItem(
        "Tôi cần đặt vé máy bay đi thành phố Hồ Chí Minh vào cuối tuần này, và nếu được "
        "thì đặt luôn khách sạn gần sân bay trong hai đêm.",
        "cau_dai",
    ),

    # --- Câu ngắn (dễ bị cắt cụt hoặc nhầm với backchannel ở tầng turn-taking) ---
    TestItem("Trời hôm nay có vẻ sắp mưa to.", "cau_ngan"),
    TestItem("Được rồi, cảm ơn.", "cau_ngan"),
]

# Giữ tương thích ngược cho code cũ import trực tiếp danh sách câu phẳng.
SENTENCES: list[str] = [item.text for item in TEST_ITEMS]

DOMAINS: list[str] = sorted({item.domain for item in TEST_ITEMS})

# Mức nhiễu mô phỏng để đo Acc dưới điều kiện suy giảm SNR (xem giới hạn ở docstring trên
# — nhiễu trắng tuyến tính, không thay được thu âm thật trong môi trường nhiễu).
NOISE_LEVELS_DB: list[float | None] = [None, 15.0, 5.0]  # None = audio sạch, không cộng nhiễu
