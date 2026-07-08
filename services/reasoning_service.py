"""Reasoning Service — phân tích/kiểm chứng một câu hỏi hoặc lựa chọn. Bản mỏng: gọi
Claude với system prompt định hướng phân tích logic (khác Planning: đây là "nghĩ", không
phải "chia bước làm")."""

from services.llm_client import complete
from services.base import BaseService

SYSTEM_PROMPT = (
    "Bạn phân tích vấn đề một cách logic: nêu giả định, rủi ro, và kết luận rõ ràng. "
    "Trả lời ngắn gọn, có cấu trúc."
)


class ReasoningService(BaseService):
    async def handle(self, request: dict) -> dict:
        question = request.get("question", "")
        analysis = await complete(SYSTEM_PROMPT, [{"role": "user", "content": question}])
        return {"analysis": analysis}
