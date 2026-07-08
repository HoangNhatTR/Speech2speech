"""Tool Service — registry các tool có thể gọi được. Bản mỏng: 1-2 tool ví dụ, đủ để
chứng minh cơ chế; thêm tool mới chỉ cần đăng ký vào TOOLS.

Request: {"name": "get_current_time", "arguments": {...}}
"""

from datetime import datetime, timezone

from services.base import BaseService


async def get_current_time(args: dict) -> dict:
    return {"utc_time": datetime.now(timezone.utc).isoformat()}


async def echo(args: dict) -> dict:
    return {"echo": args.get("text", "")}


TOOLS = {
    "get_current_time": get_current_time,
    "echo": echo,
}


class ToolService(BaseService):
    async def handle(self, request: dict) -> dict:
        name = request.get("name")
        args = request.get("arguments", {})
        tool = TOOLS.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        result = await tool(args)
        return {"result": result}
