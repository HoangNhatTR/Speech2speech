"""Vietnamese Turn-Taking Classifier Bench — bộ đầu tiên cho tiếng Việt chấm điểm
khả năng phân biệt backchannel ("dạ", "vâng ạ", "ừm"...) với ngắt lời thật của
`duplex.turn_strategies.VietnameseBackchannelTurnStartStrategy`.

Phạm vi (đọc kỹ trước khi trích dẫn số liệu): đây KHÔNG phải bản dựng lại đầy đủ
Full-Duplex-Bench gốc (benchmark đó đo trên audio thật có chồng lấn, gồm nhiều tác vụ:
pause handling, overlap handling, backchannel, smooth turn-taking...). Bench này chỉ đo
MỘT thành phần: quyết định của turn-start strategy trên chuỗi frame ASR mô phỏng
(không cần audio thật, không tốn GPU/API) — coi là bench cấp thành phần (component-level),
không phải benchmark hệ thống đầy đủ. Value thật: đây là bộ đánh giá có cấu trúc đầu
tiên cho bài toán backchannel-vs-interrupt tiếng Việt, chạy được ngay và miễn phí, dùng
làm smoke test hồi quy khi chỉnh `DEFAULT_BACKCHANNEL_WORDS`/logic phân loại.
"""
