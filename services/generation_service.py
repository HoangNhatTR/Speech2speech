"""Generation Service — tổng hợp câu trả lời cuối cùng từ context đã thu thập (kế
hoạch, phân tích, kết quả tool, lịch sử hội thoại). Bản mỏng: gọi Claude để soạn một
câu trả lời tự nhiên, mạch lạc từ context đó."""

from services.llm_client import complete
from services.base import BaseService

SYSTEM_PROMPT = (
    "Bạn tổng hợp các thông tin được cung cấp (kế hoạch, phân tích, kết quả tool, lịch "
    "sử hội thoại) thành MỘT câu trả lời tự nhiên, mạch lạc cho người dùng, tiếng Việt."
)


class GenerationService(BaseService):
    async def handle(self, request: dict) -> dict:
        context = request.get("context", "")
        reply = await complete(SYSTEM_PROMPT, [{"role": "user", "content": context}])
        return {"reply": reply}
