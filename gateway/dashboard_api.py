"""API cho Dashboard + Settings (FE tách rời độc lập trong frontend/) — quy ước "BE, FE
tách rời độc lập, phải có dashboard và settings". Router riêng, mount vào
gateway/main.py, dùng lại đúng cơ chế auth (`require_auth`) và các module đã có
(session_manager, runtime_config, eval.*) thay vì tạo cơ chế mới.

Phạm vi cố ý: /api/config CHỈ đổi được switch của Speech Service (xem
gateway/runtime_config.py::ALLOWED_KEYS) — không đổi được SERVICES_LLM_BACKEND của 7
service phía runtime.dispatcher (tiến trình riêng, xem docstring runtime_config.py).
Không có endpoint "chạy benchmark ASR WER/tool-call" vì hai benchmark đó gọi API/GPU
thật — để CLI-only (`python -m eval.run_benchmarks --with-asr`) tránh mở thêm một
đường tốn phí/quá tải có thể bị gọi lặp lại qua HTTP (xem rủi ro no-rate-limit đã ghi
trong docs/platform-architecture.md). Benchmark duplex (miễn phí, CPU thuần) thì có thể
trigger qua API vì không tốn phí và chạy dưới 1 giây.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from eval import benchmark_history, latency_report
from eval.duplex_bench import runner as duplex_bench_runner
from eventbus.client import request as bus_request
from gateway import runtime_config
from gateway.auth import require_auth
from gateway.session_manager import session_manager
from s2s.orchestrator import orchestrator_registry

router = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

_started_at = time.time()


async def _http_check(url: str, *, timeout_s: float = 2.0) -> tuple[bool, str]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if 200 <= response.status < 300:
                    return True, f"HTTP {response.status}"
                return False, f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


async def _tcp_url_check(url: str, *, timeout_s: float = 1.0) -> tuple[bool, str]:
    """Lightweight liveness for Moshi, whose official server has no health route.

    We intentionally do not open ``/api/chat`` here: the PyTorch server serializes
    inference sessions with a lock, so a readiness websocket could occupy or wait for
    the only model slot.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        return False, "invalid MOSHI_URL"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname, port), timeout=timeout_s
        )
        writer.close()
        await writer.wait_closed()
        return True, f"TCP {parsed.hostname}:{port} open"
    except Exception as exc:
        return False, str(exc)


@router.get("/status")
async def status():
    return {
        "uptime_s": round(time.time() - _started_at, 1),
        "active_sessions": session_manager.count(),
        "sessions": [
            {"session_id": s.session_id, "kind": s.kind, "created_at": s.created_at}
            for s in session_manager.list_sessions()
        ],
        "s2s_sessions": orchestrator_registry.snapshots(),
    }


@router.get("/ready")
async def ready():
    """Dependency-aware readiness for local small/medium deployment."""

    checks: dict[str, dict[str, str | bool]] = {
        "gateway": {"ok": True, "detail": "running"},
    }

    stt_backend = runtime_config.get("STT_BACKEND", "cloud")
    if stt_backend == "local":
        model_dir = Path(os.getenv("ASR_MODEL_DIR", "models/asr-vi"))
        required = ("tokens.txt", "encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx")
        missing = [name for name in required if not (model_dir / name).exists()]
        checks["asr"] = {
            "ok": not missing,
            "detail": "model files ready" if not missing else f"missing: {', '.join(missing)}",
        }
    else:
        checks["asr"] = {
            "ok": bool(os.getenv("DEEPGRAM_API_KEY")),
            "detail": "cloud key configured" if os.getenv("DEEPGRAM_API_KEY") else "missing DEEPGRAM_API_KEY",
        }

    llm_backend = runtime_config.get("LLM_BACKEND", "cloud")
    if llm_backend == "local":
        base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        ok, detail = await _http_check(f"{base}/models")
        checks["llm"] = {"ok": ok, "detail": detail}
    else:
        checks["llm"] = {
            "ok": bool(os.getenv("ANTHROPIC_API_KEY")),
            "detail": "cloud key configured" if os.getenv("ANTHROPIC_API_KEY") else "missing ANTHROPIC_API_KEY",
        }

    tts_backend = runtime_config.get("TTS_BACKEND", "cloud")
    if tts_backend == "local":
        base = os.getenv("VIENEU_SERVER_URL", "http://localhost:8100").rstrip("/")
        ok, detail = await _http_check(f"{base}/health")
        checks["tts"] = {"ok": ok, "detail": detail}
    else:
        has_tts = bool(os.getenv("ELEVENLABS_API_KEY") and os.getenv("ELEVENLABS_VOICE_ID"))
        checks["tts"] = {
            "ok": has_tts,
            "detail": "cloud keys configured" if has_tts else "missing ElevenLabs key/voice",
        }

    try:
        tool_result = await bus_request(
            "svc.tool", {"name": "echo", "arguments": {"text": "ready"}}, timeout=1.5
        )
        tool_ok = tool_result.get("result", {}).get("echo") == "ready"
        checks["eventbus_dispatcher"] = {
            "ok": tool_ok,
            "detail": "NATS request/reply ready" if tool_ok else "unexpected tool response",
        }
    except Exception as exc:
        checks["eventbus_dispatcher"] = {"ok": False, "detail": str(exc)}

    try:
        s2s_mode = runtime_config.get("S2S_MODE", "shadow")
        s2s_backend = runtime_config.get("S2S_SHADOW_BACKEND", "probe")
        if s2s_mode == "off" or s2s_backend == "probe":
            s2s_ok = True
            sidecar_detail = ""
        elif s2s_backend in {"moshi", "moshi_ws"}:
            s2s_ok, sidecar_detail = await _tcp_url_check(
                os.getenv("MOSHI_URL", "ws://127.0.0.1:8998/api/chat")
            )
        else:
            s2s_ok = False
            sidecar_detail = "unknown backend"
        checks["s2s_control_plane"] = {
            "ok": s2s_ok,
            "detail": (
                f"mode={s2s_mode}, backend={s2s_backend}"
                + (f", {sidecar_detail}" if sidecar_detail else "")
            ),
        }
    except Exception as exc:
        checks["s2s_control_plane"] = {"ok": False, "detail": str(exc)}

    is_ready = all(bool(check["ok"]) for check in checks.values())
    payload = {
        "ready": is_ready,
        "profile": os.getenv("LOCAL_PROFILE", "custom"),
        "checks": checks,
    }
    return JSONResponse(payload, status_code=200 if is_ready else 503)


@router.get("/config")
async def get_config():
    return {
        "values": runtime_config.snapshot(),
        "overridden": sorted(runtime_config.overridden_keys()),
        "allowed_keys": sorted(runtime_config.ALLOWED_KEYS),
    }


@router.put("/config")
async def put_config(overrides: dict[str, str]):
    invalid = set(overrides) - runtime_config.ALLOWED_KEYS
    if invalid:
        raise HTTPException(422, f"Key không được phép đổi runtime: {sorted(invalid)}")
    for key, value in overrides.items():
        runtime_config.set_override(key, value)
    return {"values": runtime_config.snapshot(), "overridden": sorted(runtime_config.overridden_keys())}


@router.delete("/config/{key}")
async def reset_config_key(key: str):
    if key not in runtime_config.ALLOWED_KEYS:
        raise HTTPException(422, f"Key không được phép: {key}")
    runtime_config.clear_override(key)
    return {"values": runtime_config.snapshot()}


@router.get("/metrics/latency")
async def metrics_latency():
    return {"rows": latency_report.summarize()}


@router.get("/benchmarks/history")
async def benchmarks_history(limit: int = 200):
    entries = benchmark_history.read_all()
    return {"entries": entries[-limit:]}


@router.get("/benchmarks/latest")
async def benchmarks_latest():
    return benchmark_history.latest() or {}


@router.post("/benchmarks/duplex/run")
async def run_duplex_bench():
    """Chạy Vietnamese Turn-Taking Classifier Bench (miễn phí, thuần CPU, <1s) và ghi
    vào benchmark_history — an toàn để expose qua API vì không gọi API/GPU trả phí nào,
    khác với ASR WER/tool-call (xem docstring đầu file)."""
    summary = await duplex_bench_runner.run()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "duplex_bench": {
            "accuracy": summary["accuracy"],
            "false_interrupt_rate": summary["false_interrupt_rate"],
            "missed_interrupt_rate": summary["missed_interrupt_rate"],
        },
    }
    benchmark_history.append(entry)
    return summary
