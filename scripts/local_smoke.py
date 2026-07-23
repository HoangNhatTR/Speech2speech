"""End-to-end local dependency smoke test without a browser or microphone.

This verifies the same services used by a WebRTC voice turn: Gateway readiness, NATS
request/reply and Tool Service, vLLM generation, VieNeu streaming first audio, S2S
control-plane configuration and (optionally) the dashboard frontend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp
import nats
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    latency_ms: float
    detail: str


async def _json_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> tuple[int, Any, float]:
    started = time.perf_counter()
    try:
        async with session.request(method, url, headers=headers, json=json_body) as response:
            try:
                payload = await response.json()
            except Exception:
                payload = await response.text()
            return response.status, payload, (time.perf_counter() - started) * 1000
    except Exception as exc:
        return 0, {"error": str(exc)}, (time.perf_counter() - started) * 1000


async def _check_tool() -> Check:
    started = time.perf_counter()
    nc = None
    try:
        nc = await nats.connect(os.getenv("NATS_URL", "nats://localhost:4222"), connect_timeout=2)
        request = {"name": "echo", "arguments": {"text": "speech2speech-smoke"}}
        msg = await nc.request("svc.tool", json.dumps(request).encode(), timeout=3)
        result = json.loads(msg.data)
        ok = result.get("result", {}).get("echo") == "speech2speech-smoke"
        detail = "NATS -> svc.tool -> response" if ok else f"unexpected: {result}"
    except Exception as exc:
        ok, detail = False, str(exc)
    finally:
        if nc is not None and not nc.is_closed:
            await nc.close()
    return Check("tool_request_reply", ok, (time.perf_counter() - started) * 1000, detail)


async def _check_webrtc_surface(
    session: aiohttp.ClientSession, gateway: str
) -> Check:
    """Verify the browser UI and the two-step signaling routes it depends on."""

    started = time.perf_counter()
    base = gateway.rstrip("/")
    try:
        async with session.get(f"{base}/client/") as response:
            html = await response.text()
            ui_ok = response.status == 200 and "Pipecat UI" in html
        async with session.get(f"{base}/openapi.json") as response:
            schema = await response.json()
            paths = schema.get("paths", {})
            routes_ok = "/start" in paths and "/sessions/{session_id}/{path}" in paths
        ok = ui_ok and routes_ok
        detail = "WebRTC UI + /start + session proxy" if ok else "missing UI/signaling route"
    except Exception as exc:
        ok, detail = False, str(exc)
    return Check("webrtc_surface", ok, (time.perf_counter() - started) * 1000, detail)


async def _check_tts(session: aiohttp.ClientSession) -> Check:
    started = time.perf_counter()
    base = os.getenv("VIENEU_SERVER_URL", "http://localhost:8100").rstrip("/")
    try:
        async with session.post(
            f"{base}/synthesize/stream",
            json={"text": "Xin chào, đây là kiểm tra âm thanh thời gian thực."},
        ) as response:
            first_chunk = await response.content.read(4096)
            sample_rate = response.headers.get("X-Audio-Sample-Rate")
            ok = response.status == 200 and len(first_chunk) > 0 and bool(sample_rate)
            detail = (
                f"HTTP {response.status}, first_chunk={len(first_chunk)} bytes, sample_rate={sample_rate}"
            )
    except Exception as exc:
        ok, detail = False, str(exc)
    return Check("tts_stream_first_audio", ok, (time.perf_counter() - started) * 1000, detail)


async def _check_llm(
    session: aiohttp.ClientSession, model: str, api_key: str
) -> Check:
    started = time.perf_counter()
    base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    try:
        async with session.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Chỉ trả lời đúng một từ: sẵn_sàng"}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            },
        ) as response:
            payload = await response.json()
            content = ""
            if response.status == 200:
                content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            ok = response.status == 200 and bool(content.strip())
            detail = f"HTTP {response.status}, content={content.strip()[:80]!r}"
    except Exception as exc:
        ok, detail = False, str(exc)
    return Check("llm_completion", ok, (time.perf_counter() - started) * 1000, detail)


async def run(args) -> tuple[list[Check], dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    checks: list[Check] = []
    context: dict[str, Any] = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        status, ready_payload, latency = await _json_request(
            session, "GET", f"{args.gateway.rstrip('/')}/api/ready", headers=headers
        )
        ready_ok = status == 200 and isinstance(ready_payload, dict) and ready_payload.get("ready")
        checks.append(Check("gateway_readiness", bool(ready_ok), latency, f"HTTP {status}"))
        context["readiness"] = ready_payload

        checks.append(await _check_webrtc_surface(session, args.gateway))

        status, config_payload, latency = await _json_request(
            session, "GET", f"{args.gateway.rstrip('/')}/api/config", headers=headers
        )
        values = config_payload.get("values", {}) if isinstance(config_payload, dict) else {}
        mode = values.get("S2S_MODE")
        s2s_ok = status == 200 and mode in {"off", "shadow", "ack_only", "speculative", "primary"}
        checks.append(Check("s2s_config", s2s_ok, latency, f"mode={mode}"))

        checks.append(await _check_tool())

        if not args.quick and values.get("LLM_BACKEND") == "local":
            checks.append(
                await _check_llm(
                    session,
                    values.get("VLLM_MODEL") or os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
                    os.getenv("VLLM_API_KEY", "not-needed"),
                )
            )
        if not args.quick and values.get("TTS_BACKEND") == "local":
            checks.append(await _check_tts(session))

        if args.frontend:
            started = time.perf_counter()
            try:
                async with session.get(args.frontend) as response:
                    body = await response.text()
                    ok = response.status == 200 and "<html" in body.lower()
                    detail = f"HTTP {response.status}"
            except Exception as exc:
                ok, detail = False, str(exc)
            checks.append(
                Check("frontend", ok, (time.perf_counter() - started) * 1000, detail)
            )

    return checks, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:7860")
    parser.add_argument("--frontend", default="http://localhost:5173")
    parser.add_argument("--api-key", default=os.getenv("GATEWAY_API_KEYS", "").split(",")[0])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--quick", action="store_true", help="Bỏ qua sinh LLM/TTS thật")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks, context = asyncio.run(run(args))
    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
        **context,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(
                f"{'PASS' if check.ok else 'FAIL':4} "
                f"{check.name:26} {check.latency_ms:8.1f} ms  {check.detail}"
            )
        print("PASS: local stack smoke test" if payload["ok"] else "FAIL: local stack chưa ready")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
