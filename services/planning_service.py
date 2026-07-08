"""Planning Service — chia nhỏ một yêu cầu phức tạp thành các bước cần làm. Bản mỏng:
gọi Claude với system prompt định hướng lập kế hoạch (chưa tự thực thi kế hoạch)."""

from services.llm_client import complete
from services.base import BaseService

SYSTEM_PROMPT = (
    "Bạn là bộ lập kế hoạch. Cho một yêu cầu, liệt kê ngắn gọn các bước cần làm để "
    "hoàn thành nó (mỗi bước một dòng, đánh số). Không thực hiện, chỉ lập kế hoạch."
)


class PlanningService(BaseService):
    async def handle(self, request: dict) -> dict:
        goal = request.get("goal", "")
        plan = await complete(SYSTEM_PROMPT, [{"role": "user", "content": goal}])
        return {"plan": plan}
