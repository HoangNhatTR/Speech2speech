"""Tập câu tiếng Việt tham chiếu cho eval harness (Track A trong đề xuất cải tiến).

Đây KHÔNG phải benchmark chuẩn (VLSP2020/2023, Common Voice, FLEURS-vi) — những bộ đó
cần đăng ký/tải qua kênh chính thức, chưa tải được trong lần build này. Đây là tập nhỏ
tự viết để eval harness có dữ liệu thật ngay hôm nay: dùng gTTS (Google Translate TTS,
miễn phí, không cần API key) tổng hợp audio từ câu văn bản đã biết trước, rồi đo WER của
ASR trên chính audio đó. Nhược điểm cần biết: audio TTS sạch hơn giọng người thật nhiều
(không nhiễu, không nói lắp, phát âm chuẩn) — WER đo được ở đây là "ideal case", KHÔNG
đại diện cho WER trên giọng nói thật ngoài đời. Coi đây là smoke test hồi quy (so sánh
model A vs B trên cùng điều kiện), không phải con số WER cuối cùng để công bố.

Nâng cấp khuyến nghị (chưa làm): thay bằng VIVOS/VLSP2020 test set khi tải được, để có
WER thật trên giọng người.
"""

SENTENCES = [
    "Xin chào, hôm nay thời tiết rất đẹp.",
    "Bạn có thể giúp tôi đặt lịch hẹn vào ngày mai không?",
    "Tôi muốn kiểm tra số dư tài khoản ngân hàng của mình.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Xin lỗi, tôi không nghe rõ, bạn có thể nhắc lại không?",
    "Cảm ơn bạn rất nhiều vì đã giúp đỡ tôi.",
    "Bây giờ là mấy giờ rồi nhỉ?",
    "Món phở bò là một trong những món ăn nổi tiếng của Việt Nam.",
    "Tôi cần đặt vé máy bay đi thành phố Hồ Chí Minh vào cuối tuần này.",
    "Trời hôm nay có vẻ sắp mưa to.",
    "Anh có thể chuyển tiền giúp tôi được không?",
    "Chúc bạn một ngày làm việc hiệu quả và vui vẻ.",
]
