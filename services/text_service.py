"""Text Service — hội thoại thuần văn bản (không audio). Bản mỏng: gọi Claude trực
tiếp với một system prompt hội thoại chung."""

from services.llm_client import complete
from services.base import BaseService

SYSTEM_PROMPT = "Bạn là trợ lý hội thoại bằng văn bản, trả lời ngắn gọn, rõ ràng, tiếng Việt."


class TextService(BaseService):
    async def handle(self, request: dict) -> dict:
        content = request.get("content", "")
        reply = await complete(SYSTEM_PROMPT, [{"role": "user", "content": content}])
        return {"reply": reply}
