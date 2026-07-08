"""Vision Service — hiểu hình ảnh. Bản mỏng: gọi LLM vision qua services/llm_client.py
(provider-agnostic — xem SERVICES_LLM_BACKEND).

Request: {"image_base64": "...", "media_type": "image/jpeg", "prompt": "..."}
"""

from services.base import BaseService
from services.llm_client import complete_vision

SYSTEM_PROMPT = "Bạn mô tả và trả lời câu hỏi về hình ảnh một cách chính xác, ngắn gọn, tiếng Việt."


class VisionService(BaseService):
    async def handle(self, request: dict) -> dict:
        reply = await complete_vision(
            SYSTEM_PROMPT,
            prompt=request.get("prompt", "Mô tả hình ảnh này."),
            image_base64=request["image_base64"],
            media_type=request.get("media_type", "image/jpeg"),
        )
        return {"reply": reply}
