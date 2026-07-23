"""Runtime Config — các switch không nhạy cảm (chọn backend, chọn model, bật/tắt tính
năng) có thể đổi khi Gateway đang chạy, để Dashboard/Settings (FE) chỉnh qua API thay vì
sửa .env + restart thủ công. KHÔNG BAO GIỜ chứa API key/secret — những cái đó chỉ đọc từ
os.getenv trực tiếp trong bot.py/services, không đi qua store này và không expose qua
API (xem _ALLOWED_KEYS: chỉ liệt kê tên backend/model, không có key nào chứa "KEY" hay
"SECRET").

Cách hoạt động: bot.py và duplex/emotion.py đọc qua get(key, default) thay vì
os.getenv(key, default) trực tiếp — get() ưu tiên override runtime (nếu Settings đã đổi)
trước, rồi mới rơi về os.getenv (giá trị từ .env). Mỗi phiên WebRTC/voice mới gọi
build_stt/build_llm/build_tts lại từ đầu (xem gateway/main.py::offer), nên đổi setting ở
đây có hiệu lực ngay cho PHIÊN MỚI — không ảnh hưởng phiên đang chạy dở (hot-swap theo
phiên, không phải live-migrate phiên đang mở).

In-memory, một tiến trình — giống gateway/session_manager.py, mất khi restart Gateway.
Cần nhiều instance Gateway thì chuyển sang Redis giống khuyến nghị trong
session_manager.py.
"""

import os

# Chỉ những key này được phép đổi runtime — cố ý không gồm bất kỳ *_API_KEY/*_URL trỏ
# tới dịch vụ nội bộ nhạy cảm nào.
#
# CHỈ áp dụng cho Speech Service (bot.py, chạy TRONG tiến trình Gateway — xem
# docs/platform-architecture.md). KHÔNG bao gồm SERVICES_LLM_BACKEND: 7 service
# Vision/Text/Tool/Memory/Planning/Reasoning/Generation chạy trong tiến trình
# runtime.dispatcher RIÊNG (Ray actors) — store in-memory này không với tới đó được.
# Muốn đổi SERVICES_LLM_BACKEND runtime (không restart dispatcher) cần một cơ chế khác
# (vd publish message cấu hình qua NATS để dispatcher tự áp dụng) — chưa làm, vẫn phải
# sửa .env + restart `python -m runtime.dispatcher` như cũ.
ALLOWED_KEYS: set[str] = {
    "STT_BACKEND",
    "LLM_BACKEND",
    "TTS_BACKEND",
    "DUPLEX_BACKCHANNEL_FILTER",
    "DUPLEX_BARGE_IN_DELAY_MS",
    "EMOTION_BACKEND",
    "ANTHROPIC_MODEL",
    "DEEPGRAM_LANGUAGE",
    "ELEVENLABS_MODEL",
    "VIENEU_STREAMING",
    "VLLM_MODEL",
    "S2S_MODE",
    "S2S_SHADOW_BACKEND",
    "S2S_AUDIO_RING_MS",
    "S2S_SUBSCRIBER_QUEUE_FRAMES",
}
# LƯU Ý: VIENEU_VOICE/VIENEU_REF_AUDIO/VIENEU_DEVICE (selfhost/tts_server.py) KHÔNG nằm
# trong danh sách này — cùng lý do với SERVICES_LLM_BACKEND ở trên: TTS server chạy
# trong tiến trình .venv-tts RIÊNG, đọc os.getenv() của chính tiến trình đó, không phải
# tiến trình Gateway — override ở đây sẽ không có tác dụng. Muốn đổi VIENEU_VOICE runtime
# cần restart TTS server (hoặc thêm cơ chế cấu hình riêng cho tiến trình đó).

_overrides: dict[str, str] = {}

_DEFAULT_VALUES: dict[str, str] = {
    "S2S_MODE": "shadow",
    "S2S_SHADOW_BACKEND": "probe",
    "S2S_AUDIO_RING_MS": "30000",
    "S2S_SUBSCRIBER_QUEUE_FRAMES": "128",
}


def get(key: str, default: str) -> str:
    if key in _overrides:
        return _overrides[key]
    return os.getenv(key, default)


def set_override(key: str, value: str) -> None:
    if key not in ALLOWED_KEYS:
        raise KeyError(f"'{key}' không nằm trong danh sách setting được phép đổi runtime")
    _overrides[key] = value


def clear_override(key: str) -> None:
    _overrides.pop(key, None)


def snapshot() -> dict[str, str]:
    """Giá trị đang hiệu lực cho mọi key được phép đổi (override nếu có, không thì giá
    trị .env, không thì rỗng) — dùng cho GET /api/config. Không bao giờ trả secret vì
    ALLOWED_KEYS không chứa key bí mật nào."""
    return {key: get(key, _DEFAULT_VALUES.get(key, "")) for key in sorted(ALLOWED_KEYS)}


def overridden_keys() -> set[str]:
    return set(_overrides.keys())
