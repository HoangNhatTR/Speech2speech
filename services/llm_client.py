"""Wrapper provider-agnostic dùng chung cho text/vision/planning/reasoning/generation
service — Track B trong đề xuất cải tiến (docs/platform-architecture.md). Trước đây
file này (services/anthropic_client.py) hard-code Claude; giờ chuyển qua
SERVICES_LLM_BACKEND=cloud|local để test model nhỏ hơn tự host cho cả 5 service này,
đúng mẫu đã dùng cho Speech Service trong bot.py::build_llm (nhưng dùng switch riêng —
2 pipeline độc lập, đổi model cho voice không bắt buộc đổi luôn model cho Vision/
Planning/...).

cloud  -> Claude qua Anthropic API (mặc định, đã verify chạy tốt).
local  -> bất kỳ endpoint tương thích OpenAI (vd vLLM serve) — CHƯA verify chạy (cùng
          giới hạn với bot.py::build_llm: vLLM cần WSL2/Linux + GPU đủ VRAM).
"""

import os
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

_anthropic_client: Optional[AsyncAnthropic] = None
_openai_client: Optional[AsyncOpenAI] = None


def backend() -> str:
    return os.getenv("SERVICES_LLM_BACKEND", "cloud")


def _get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            base_url=os.environ["SERVICES_VLLM_BASE_URL"],
            api_key=os.getenv("SERVICES_VLLM_API_KEY", "not-needed"),
        )
    return _openai_client


def default_model() -> str:
    if backend() == "local":
        return os.getenv("SERVICES_VLLM_MODEL", "Qwen/Qwen3-8B-Instruct")
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


async def complete(system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """Gọi LLM (text-only), trả về phần text của phản hồi."""
    if backend() == "local":
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=default_model(),
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.choices[0].message.content or ""

    client = _get_anthropic_client()
    response = await client.messages.create(
        model=default_model(),
        system=system,
        max_tokens=max_tokens,
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


async def complete_vision(
    system: str,
    prompt: str,
    image_base64: str,
    media_type: str = "image/jpeg",
    max_tokens: int = 1024,
) -> str:
    """Gọi LLM vision (ảnh + câu hỏi), trả về text."""
    if backend() == "local":
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=default_model(),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content or ""

    client = _get_anthropic_client()
    response = await client.messages.create(
        model=default_model(),
        system=system,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_base64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
