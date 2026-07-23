"""One-command local stack manager for small and medium deployments.

All mutable state stays under this repository: logs in ``logs/`` and the PID manifest
in ``.runtime/``. The manager reuses already healthy services but only stops processes
that it started and recorded itself.

Examples:
    .venv/bin/python -m scripts.local_stack doctor --profile small
    .venv/bin/python -m scripts.local_stack start --profile small
    .venv/bin/python -m scripts.local_stack status
    .venv/bin/python -m scripts.local_stack stop
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
MANIFEST_PATH = RUNTIME_DIR / "local_stack.json"
LOG_DIR = ROOT / "logs"
PROFILE_DIR = ROOT / "config" / "profiles"


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    name: str
    command: list[str]
    log_path: str


def _load_environment(profile: str) -> dict[str, str]:
    profile_path = PROFILE_DIR / f"{profile}.env"
    if not profile_path.exists():
        raise ValueError(f"Không có profile: {profile_path}")

    merged: dict[str, str] = {}
    # Profile provides safe defaults. User .env overrides it; shell environment has
    # highest priority for CI/temporary tuning.
    for source in (dotenv_values(profile_path), dotenv_values(ROOT / ".env")):
        for key, value in source.items():
            if value is not None:
                merged[key] = value
    merged.update(os.environ)
    merged["LOCAL_PROFILE"] = profile
    merged["PYTHONUNBUFFERED"] = "1"
    return merged


def _tcp_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _moshi_endpoint(env: dict[str, str]) -> tuple[str, int]:
    parsed = urlparse(env.get("MOSHI_URL", "ws://127.0.0.1:8998/api/chat"))
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("MOSHI_URL phải là websocket URL hợp lệ")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("--with-moshi chỉ quản lý sidecar loopback local")
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "wss" else 80)


def _python_has_module(python: Path, module: str) -> bool:
    if not python.exists():
        return False
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.util,sys;"
                    f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
                ),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _http_json(url: str, timeout: float = 2.0) -> tuple[bool, dict[str, Any] | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
            try:
                return 200 <= response.status < 300, json.loads(raw)
            except json.JSONDecodeError:
                return 200 <= response.status < 300, raw.decode(errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body: dict[str, Any] | str = json.loads(exc.read())
        except Exception:
            body = str(exc)
        return False, body
    except Exception as exc:
        return False, str(exc)


async def _dispatcher_ready(timeout: float = 1.0) -> bool:
    try:
        import nats

        nc = await nats.connect(os.getenv("NATS_URL", "nats://localhost:4222"), connect_timeout=timeout)
        payload = json.dumps({"name": "echo", "arguments": {"text": "local-stack"}}).encode()
        msg = await nc.request("svc.tool", payload, timeout=timeout)
        await nc.close()
        data = json.loads(msg.data)
        return data.get("result", {}).get("echo") == "local-stack"
    except Exception:
        return False


def _wait_until(
    name: str, predicate, timeout_s: float, *, process_pid: int | None = None
) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if predicate():
            return
        if process_pid is not None and not _pid_alive(process_pid):
            raise RuntimeError(
                f"{name} đã thoát trước khi ready (PID {process_pid}); xem log tương ứng"
            )
        time.sleep(0.5)
    raise RuntimeError(f"{name} chưa ready sau {timeout_s:.0f}s")


def _read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"managed": {}, "external": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"managed": {}, "external": []}


def _write_manifest(manifest: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_belongs_to_repo(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except OSError:
        return False
    return str(ROOT) in cmdline or cmdline.startswith(".venv/") or "nats-server" in cmdline


def _spawn(spec: ServiceSpec, env: dict[str, str], manifest: dict[str, Any]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / spec.log_path
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            spec.command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    manifest["managed"][spec.name] = {
        "pid": process.pid,
        "command": spec.command,
        "log_path": spec.log_path,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest)
    return process.pid


def _mark_external(manifest: dict[str, Any], name: str) -> None:
    if name not in manifest["external"]:
        manifest["external"].append(name)
    _write_manifest(manifest)


def _doctor(
    profile: str,
    *,
    print_report: bool = True,
    require_frontend: bool = True,
    require_moshi: bool = False,
) -> bool:
    env = _load_environment(profile)
    rows: list[tuple[str, bool, str]] = []

    rows.append(("main_venv", (ROOT / ".venv/bin/python").exists(), ".venv/bin/python"))
    rows.append(("env_file", (ROOT / ".env").exists(), ".env"))
    nats_ok = _tcp_open(4222) or (ROOT / "bin/nats-server").exists()
    rows.append(("nats", nats_ok, "port 4222 hoặc bin/nats-server"))

    if env.get("STT_BACKEND", "cloud") == "local":
        model_dir = ROOT / env.get("ASR_MODEL_DIR", "models/asr-vi")
        required = ("tokens.txt", "encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx")
        ok = all((model_dir / name).exists() for name in required)
        rows.append(("asr_local", ok, str(model_dir.relative_to(ROOT)) if model_dir.is_relative_to(ROOT) else str(model_dir)))
    else:
        rows.append(("asr_cloud_key", bool(env.get("DEEPGRAM_API_KEY")), "DEEPGRAM_API_KEY"))

    if env.get("LLM_BACKEND", "cloud") == "local":
        ok = _tcp_open(8000) or (ROOT / ".venv-vllm/bin/vllm").exists()
        rows.append(("vllm", ok, "port 8000 hoặc .venv-vllm/bin/vllm"))
    else:
        rows.append(("llm_cloud_key", bool(env.get("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY"))

    if env.get("TTS_BACKEND", "cloud") == "local":
        ok = _tcp_open(8100) or (ROOT / ".venv-tts/bin/python").exists()
        rows.append(("tts_local", ok, "port 8100 hoặc .venv-tts/bin/python"))
    else:
        ok = bool(env.get("ELEVENLABS_API_KEY") and env.get("ELEVENLABS_VOICE_ID"))
        rows.append(("tts_cloud_keys", ok, "ELEVENLABS_API_KEY + VOICE_ID"))

    if require_frontend:
        rows.append(("frontend", (ROOT / "frontend/node_modules").exists(), "frontend/node_modules"))
    if require_moshi:
        s2s_python = ROOT / ".venv-s2s/bin/python"
        rows.append(
            (
                "moshi_runtime",
                _python_has_module(s2s_python, "moshi"),
                ".venv-s2s (moshi importable)",
            )
        )
    rows.append(("profile", True, str(PROFILE_DIR / f"{profile}.env")))

    if print_report:
        print(f"Local profile: {profile}")
        for name, ok, detail in rows:
            print(f"  {'OK' if ok else 'FAIL':4}  {name:20} {detail}")
    return all(ok for _, ok, _ in rows)


def _start(
    profile: str, *, skip_frontend: bool, with_monitoring: bool, with_moshi: bool
) -> None:
    if not _doctor(
        profile,
        require_frontend=not skip_frontend,
        require_moshi=with_moshi,
    ):
        raise SystemExit("Doctor chưa đạt; sửa các mục FAIL trước khi start.")

    old = _read_manifest()
    alive_managed = {
        name: item for name, item in old.get("managed", {}).items() if _pid_alive(int(item["pid"]))
    }
    if alive_managed:
        raise SystemExit("Local stack đã có process managed đang chạy; dùng status hoặc stop trước.")

    env = _load_environment(profile)
    if with_moshi:
        env["S2S_MODE"] = "shadow"
        env["S2S_SHADOW_BACKEND"] = "moshi_ws"
        # Keep every downloaded checkpoint/cache under this repository.
        env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    manifest: dict[str, Any] = {
        "profile": profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skip_frontend": skip_frontend,
        "with_moshi": with_moshi,
        "managed": {},
        "external": [],
    }
    _write_manifest(manifest)

    try:
        if _tcp_open(4222):
            _mark_external(manifest, "nats")
        else:
            _spawn(
                ServiceSpec(
                    "nats",
                    [str(ROOT / "bin/nats-server"), "-js", "-sd", str(ROOT / "data/nats"), "-p", "4222", "-m", "8222"],
                    "logs/local_nats.log",
                ),
                env,
                manifest,
            )
            _wait_until("NATS", lambda: _tcp_open(4222), 10)

        if env.get("TTS_BACKEND", "cloud") == "local":
            if _tcp_open(8100):
                _mark_external(manifest, "tts")
            else:
                _spawn(
                    ServiceSpec(
                        "tts",
                        [str(ROOT / ".venv-tts/bin/python"), str(ROOT / "selfhost/tts_server.py")],
                        "logs/local_tts.log",
                    ),
                    env,
                    manifest,
                )
                _wait_until("TTS", lambda: _http_json("http://127.0.0.1:8100/health")[0], 300)

        if env.get("LLM_BACKEND", "cloud") == "local":
            if _tcp_open(8000):
                _mark_external(manifest, "vllm")
            else:
                command = [
                    str(ROOT / ".venv-vllm/bin/vllm"),
                    "serve",
                    env.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                    "--reasoning-parser",
                    "qwen3",
                    "--default-chat-template-kwargs",
                    '{"enable_thinking": false}',
                    "--gpu-memory-utilization",
                    env.get("VLLM_GPU_MEMORY_UTILIZATION", "0.15"),
                    "--max-model-len",
                    env.get("VLLM_MAX_MODEL_LEN", "4096"),
                    "--max-num-seqs",
                    env.get("VLLM_MAX_NUM_SEQS", "4"),
                ]
                _spawn(ServiceSpec("vllm", command, "logs/local_vllm.log"), env, manifest)
                _wait_until("vLLM", lambda: _http_json("http://127.0.0.1:8000/v1/models", 3)[0], 1200)

        if with_moshi:
            moshi_host, moshi_port = _moshi_endpoint(env)
            if _tcp_open(moshi_port, moshi_host):
                _mark_external(manifest, "moshi")
            else:
                command = [
                    str(ROOT / ".venv-s2s/bin/python"),
                    "-m",
                    "moshi.server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(moshi_port),
                    "--static",
                    "none",
                    "--hf-repo",
                    env.get("MOSHI_HF_REPO", "kyutai/moshika-pytorch-bf16"),
                    "--device",
                    env.get("MOSHI_DEVICE", "cuda"),
                ]
                if env.get("MOSHI_DTYPE", "bf16").lower() in {"fp16", "float16"}:
                    command.append("--half")
                moshi_pid = _spawn(
                    ServiceSpec("moshi", command, "logs/local_moshi.log"), env, manifest
                )
                _wait_until(
                    "Moshi",
                    lambda: _tcp_open(moshi_port, moshi_host),
                    float(env.get("MOSHI_START_TIMEOUT_S", "1800")),
                    process_pid=moshi_pid,
                )

        os.environ.update(env)
        if asyncio.run(_dispatcher_ready()):
            _mark_external(manifest, "dispatcher")
        else:
            dispatcher_module = (
                "runtime.dispatcher"
                if env.get("RUNTIME_DISPATCHER") == "ray"
                else "runtime.local_dispatcher"
            )
            _spawn(
                ServiceSpec(
                    "dispatcher",
                    [str(ROOT / ".venv/bin/python"), "-m", dispatcher_module],
                    "logs/local_dispatcher.log",
                ),
                env,
                manifest,
            )
            _wait_until("Dispatcher", lambda: asyncio.run(_dispatcher_ready()), 30)

        if with_monitoring:
            _spawn(
                ServiceSpec(
                    "monitoring",
                    [str(ROOT / ".venv/bin/python"), "-m", "services.monitoring_service"],
                    "logs/local_monitoring.log",
                ),
                env,
                manifest,
            )

        if _tcp_open(7860):
            _mark_external(manifest, "gateway")
        else:
            _spawn(
                ServiceSpec(
                    "gateway",
                    [str(ROOT / ".venv/bin/python"), "-m", "gateway.main"],
                    "logs/local_gateway.log",
                ),
                env,
                manifest,
            )
            _wait_until("Gateway", lambda: _http_json("http://127.0.0.1:7860/api/status")[0], 60)

        if not skip_frontend:
            if _tcp_open(5173):
                _mark_external(manifest, "frontend")
            else:
                _spawn(
                    ServiceSpec(
                        "frontend",
                        ["npm", "--prefix", str(ROOT / "frontend"), "run", "dev", "--", "--host", "127.0.0.1"],
                        "logs/local_frontend.log",
                    ),
                    env,
                    manifest,
                )
                _wait_until("Frontend", lambda: _http_json("http://127.0.0.1:5173", 2)[0], 60)

        if with_moshi:
            config_ok, config_payload = _http_json("http://127.0.0.1:7860/api/config")
            values = config_payload.get("values", {}) if isinstance(config_payload, dict) else {}
            if not config_ok or values.get("S2S_SHADOW_BACKEND") != "moshi_ws":
                raise RuntimeError(
                    "Gateway không chạy với S2S_SHADOW_BACKEND=moshi_ws; "
                    "hãy dừng gateway external rồi start lại bằng local_stack."
                )
    except Exception:
        _stop(quiet=True)
        raise

    print("Local stack ready.")
    print("  Voice:     http://localhost:7860/client")
    print("  Readiness: http://localhost:7860/api/ready")
    if not skip_frontend:
        print("  Dashboard: http://localhost:5173")
    if with_moshi:
        print(f"  Moshi:     {env['MOSHI_URL']} (shadow only)")
    print("  Smoke:     .venv/bin/python -m scripts.local_smoke")


def _stop(*, quiet: bool = False) -> None:
    manifest = _read_manifest()
    managed = manifest.get("managed", {})
    for name, item in reversed(list(managed.items())):
        pid = int(item["pid"])
        if not _pid_alive(pid):
            continue
        if not _pid_belongs_to_repo(pid):
            if not quiet:
                print(f"SKIP {name}: PID {pid} không còn thuộc repo")
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
            if not quiet:
                print(f"TERM {name} (PID {pid})")
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if not any(_pid_alive(int(item["pid"])) for item in managed.values()):
            break
        time.sleep(0.2)

    for name, item in managed.items():
        pid = int(item["pid"])
        if _pid_alive(pid) and _pid_belongs_to_repo(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
                if not quiet:
                    print(f"KILL {name} (PID {pid})")
            except (ProcessLookupError, PermissionError):
                pass
    MANIFEST_PATH.unlink(missing_ok=True)
    if not quiet:
        print("Đã dừng toàn bộ process do local_stack quản lý; external services được giữ nguyên.")


def _status() -> bool:
    manifest = _read_manifest()
    profile = manifest.get("profile", "small")
    env = _load_environment(profile)
    print(f"Profile: {manifest.get('profile', 'none')}")
    for name, item in manifest.get("managed", {}).items():
        pid = int(item["pid"])
        print(f"  {'UP' if _pid_alive(pid) else 'DOWN':4} managed {name:12} pid={pid}")
    for name in manifest.get("external", []):
        print(f"  EXT  external {name}")

    checks: dict[str, bool] = {
        "nats": _tcp_open(4222),
        "gateway": _http_json("http://127.0.0.1:7860/api/status")[0],
        "dispatcher": asyncio.run(_dispatcher_ready()),
    }
    if env.get("LLM_BACKEND", "cloud") == "local":
        checks["vllm"] = _http_json("http://127.0.0.1:8000/v1/models")[0]
    if env.get("TTS_BACKEND", "cloud") == "local":
        checks["tts"] = _http_json("http://127.0.0.1:8100/health")[0]
    if manifest.get("with_moshi", False):
        try:
            moshi_host, moshi_port = _moshi_endpoint(env)
            checks["moshi"] = _tcp_open(moshi_port, moshi_host)
        except ValueError:
            checks["moshi"] = False
    if not manifest.get("skip_frontend", False):
        checks["frontend"] = _http_json("http://127.0.0.1:5173")[0]
    print("Health:")
    for name, ok in checks.items():
        print(f"  {'OK' if ok else 'DOWN':4} {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--profile", choices=("small", "medium"), default="small")
    doctor.add_argument("--skip-frontend", action="store_true")
    doctor.add_argument("--with-moshi", action="store_true")

    start = subparsers.add_parser("start")
    start.add_argument("--profile", choices=("small", "medium"), default="small")
    start.add_argument("--skip-frontend", action="store_true")
    start.add_argument("--with-monitoring", action="store_true")
    start.add_argument("--with-moshi", action="store_true")

    subparsers.add_parser("status")
    subparsers.add_parser("stop")
    args = parser.parse_args()

    if args.command == "doctor":
        raise SystemExit(
            0
            if _doctor(
                args.profile,
                require_frontend=not args.skip_frontend,
                require_moshi=args.with_moshi,
            )
            else 1
        )
    if args.command == "start":
        _start(
            args.profile,
            skip_frontend=args.skip_frontend,
            with_monitoring=args.with_monitoring,
            with_moshi=args.with_moshi,
        )
        return
    if args.command == "status":
        raise SystemExit(0 if _status() else 1)
    if args.command == "stop":
        _stop()


if __name__ == "__main__":
    main()
