"""Official Kyutai Moshi WebSocket adapter for the experimental shadow path.

The model remains in ``.venv-s2s`` and runs with ``python -m moshi.server``. This
Gateway-side adapter implements Kyutai's wire protocol only:

* connect to ``/api/chat`` and require the binary ``0x00`` handshake;
* resample incoming PCM16 continuously to mono 24 kHz;
* encode Opus with ``sphn`` and send binary ``0x01 + opus`` messages;
* observe returned ``0x01`` audio and ``0x02`` inner-monologue text.

Returned model audio is deliberately counted but never committed. Text is emitted as
medium-risk ``EXPLANATION`` proposals with original turn provenance, so SegmentPolicy
drops it in shadow and refuses to speculate with it in every other current mode.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp
import numpy as np
from loguru import logger

from s2s.backend import AudioTokenS2SBackend
from s2s.events import AudioChunk, FastProposal, RiskLevel, SegmentKind
from s2s.orchestrator import OrchestratorSnapshot

MOSHI_SAMPLE_RATE = 24_000
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_audio_runtime():
    try:
        import soxr
        import sphn
    except ImportError as exc:
        raise RuntimeError(
            "Backend moshi_ws cần sphn và soxr trong .venv chính; "
            "chạy pip install -r requirements.txt"
        ) from exc
    return sphn, soxr


def _validate_url(url: str, *, allow_remote: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("MOSHI_URL phải là ws:// hoặc wss:// hợp lệ")
    if parsed.hostname not in _LOCAL_HOSTS and not allow_remote:
        raise ValueError(
            "MOSHI_URL không phải loopback. Đặt MOSHI_ALLOW_REMOTE=true chỉ khi đã "
            "chấp thuận gửi audio tới sidecar từ xa."
        )


class MoshiWebSocketBackend(AudioTokenS2SBackend):
    """Non-blocking observer of an official ``moshi.server`` session."""

    name = "moshi_ws"

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:8998/api/chat",
        connect_timeout_s: float = 1.5,
        allow_remote: bool = False,
        proposal_queue_size: int = 8,
    ) -> None:
        _validate_url(url, allow_remote=allow_remote)
        if connect_timeout_s <= 0 or proposal_queue_size <= 0:
            raise ValueError("timeout và proposal_queue_size phải lớn hơn 0")
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self._session_id = ""
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task | None = None
        self._opus_writer: Any = None
        self._resampler: Any = None
        self._resampler_input_rate: int | None = None
        self._proposals: asyncio.Queue[FastProposal] = asyncio.Queue(
            maxsize=proposal_queue_size
        )
        self._text_buffer = ""
        self._text_snapshot: OrchestratorSnapshot | None = None
        self._last_snapshot: OrchestratorSnapshot | None = None
        self._connected = False
        self._closed = False
        self._first_input_at_ns: int | None = None
        self._first_audio_at_ns: int | None = None
        self._first_text_at_ns: int | None = None
        self._input_frames = 0
        self._input_audio_ms = 0.0
        self._opus_messages_sent = 0
        self._opus_bytes_sent = 0
        self._output_audio_messages = 0
        self._output_audio_bytes = 0
        self._output_text_chars = 0
        self._sequence_gaps = 0
        self._last_sequence: int | None = None
        self._dropped_proposals = 0
        self._last_error = ""

    @classmethod
    def from_env(cls) -> "MoshiWebSocketBackend":
        try:
            timeout = float(os.getenv("MOSHI_CONNECT_TIMEOUT_S", "1.5"))
            queue_size = int(os.getenv("MOSHI_PROPOSAL_QUEUE_SIZE", "8"))
        except ValueError as exc:
            raise ValueError("MOSHI_CONNECT_TIMEOUT_S/QUEUE_SIZE không hợp lệ") from exc
        return cls(
            url=os.getenv("MOSHI_URL", "ws://127.0.0.1:8998/api/chat"),
            connect_timeout_s=timeout,
            allow_remote=_bool_env("MOSHI_ALLOW_REMOTE"),
            proposal_queue_size=queue_size,
        )

    async def start(self, session_id: str) -> None:
        if self._connected:
            return
        self._session_id = session_id
        self._closed = False
        sphn, _ = _load_audio_runtime()
        timeout = aiohttp.ClientTimeout(total=None, connect=self.connect_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        try:
            self._ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self.url,
                    heartbeat=20,
                    max_msg_size=16 * 1024 * 1024,
                ),
                timeout=self.connect_timeout_s,
            )
            handshake = await asyncio.wait_for(
                self._ws.receive(), timeout=self.connect_timeout_s
            )
            if (
                handshake.type is not aiohttp.WSMsgType.BINARY
                or handshake.data != b"\x00"
            ):
                raise RuntimeError("Moshi handshake không phải binary 0x00")
            self._opus_writer = sphn.OpusStreamWriter(MOSHI_SAMPLE_RATE)
            self._connected = True
            self._receive_task = asyncio.create_task(
                self._receive_loop(), name=f"moshi-recv-{session_id}"
            )
            logger.info(f"[s2s] Moshi shadow đã nối {self.url}")
        except Exception as exc:
            self._last_error = str(exc)
            await self.stop()
            raise RuntimeError(f"Không nối được Moshi sidecar {self.url}: {exc}") from exc

    async def process_audio(
        self, chunk: AudioChunk, snapshot: OrchestratorSnapshot
    ) -> list[FastProposal]:
        self._input_frames += 1
        self._input_audio_ms += chunk.duration_ms
        self._last_snapshot = snapshot
        if self._last_sequence is not None and chunk.sequence != self._last_sequence + 1:
            self._sequence_gaps += max(1, chunk.sequence - self._last_sequence - 1)
        self._last_sequence = chunk.sequence

        ws = self._ws
        if self._connected and ws is not None and not ws.closed:
            try:
                pcm = self._to_mono_float32(chunk)
                pcm = self._resample(pcm, chunk.sample_rate)
                if pcm.size:
                    opus = self._opus_writer.append_pcm(pcm)
                    if len(opus) > 0:
                        if self._first_input_at_ns is None:
                            self._first_input_at_ns = time.monotonic_ns()
                        await ws.send_bytes(b"\x01" + opus)
                        self._opus_messages_sent += 1
                        self._opus_bytes_sent += len(opus)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                self._connected = False
                logger.warning(f"[s2s] Moshi send lỗi; shadow bị vô hiệu hóa: {exc}")

        return self._drain_proposals()

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        receive_task = self._receive_task
        self._receive_task = None
        if receive_task is not None and receive_task is not current:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if message.type is aiohttp.WSMsgType.BINARY:
                    data = message.data
                    if not isinstance(data, bytes) or not data:
                        continue
                    if data[0] == 1:
                        self._observe_audio(data[1:])
                    elif data[0] == 2:
                        self._observe_text(data[1:].decode("utf-8", errors="replace"))
                elif message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"[s2s] Moshi receive lỗi; anchor tiếp tục: {exc}")
        finally:
            self._connected = False

    def _observe_audio(self, payload: bytes) -> None:
        self._output_audio_messages += 1
        self._output_audio_bytes += len(payload)
        if self._first_audio_at_ns is None:
            self._first_audio_at_ns = time.monotonic_ns()

    def _observe_text(self, token: str) -> None:
        if not token:
            return
        if self._first_text_at_ns is None:
            self._first_text_at_ns = time.monotonic_ns()
        if not self._text_buffer:
            self._text_snapshot = self._last_snapshot
        self._text_buffer += token
        self._output_text_chars += len(token)
        stripped = self._text_buffer.strip()
        if len(stripped) >= 160 or stripped.endswith((".", "?", "!", "\n")):
            self._queue_text_proposal(stripped)
            self._text_buffer = ""
            self._text_snapshot = None

    def _queue_text_proposal(self, text: str) -> None:
        if not text:
            return
        state = self._text_snapshot
        proposal = FastProposal(
            text=text,
            kind=SegmentKind.EXPLANATION,
            risk=RiskLevel.MEDIUM,
            turn_id=state.turn_id if state else None,
            revision_id=state.revision_id if state else None,
            generation_id=state.generation_id if state else None,
        )
        if self._proposals.full():
            try:
                self._proposals.get_nowait()
                self._dropped_proposals += 1
            except asyncio.QueueEmpty:
                pass
        self._proposals.put_nowait(proposal)

    def _drain_proposals(self) -> list[FastProposal]:
        proposals: list[FastProposal] = []
        while True:
            try:
                proposals.append(self._proposals.get_nowait())
            except asyncio.QueueEmpty:
                return proposals

    def _to_mono_float32(self, chunk: AudioChunk) -> np.ndarray:
        pcm16 = np.frombuffer(chunk.audio, dtype="<i2")
        usable = pcm16.size - (pcm16.size % chunk.num_channels)
        if usable <= 0:
            return np.empty(0, dtype=np.float32)
        channels = pcm16[:usable].reshape(-1, chunk.num_channels).astype(np.float32)
        mono = channels.mean(axis=1) if chunk.num_channels > 1 else channels[:, 0]
        return mono / 32768.0

    def _resample(self, pcm: np.ndarray, input_rate: int) -> np.ndarray:
        if input_rate == MOSHI_SAMPLE_RATE:
            self._resampler = None
            self._resampler_input_rate = input_rate
            return pcm
        if self._resampler is None or self._resampler_input_rate != input_rate:
            _, soxr = _load_audio_runtime()
            self._resampler = soxr.ResampleStream(
                input_rate,
                MOSHI_SAMPLE_RATE,
                1,
                dtype="float32",
                quality="HQ",
            )
            self._resampler_input_rate = input_rate
        return self._resampler.resample_chunk(pcm, last=False)

    def stats(self) -> dict[str, int | float | str | bool]:
        def latency_ms(at_ns: int | None) -> float:
            if at_ns is None or self._first_input_at_ns is None:
                return 0.0
            return round((at_ns - self._first_input_at_ns) / 1_000_000, 3)

        return {
            "backend": self.name,
            "connected": self._connected,
            "url": self.url,
            "input_frames": self._input_frames,
            "input_audio_ms": round(self._input_audio_ms, 3),
            "opus_messages_sent": self._opus_messages_sent,
            "opus_bytes_sent": self._opus_bytes_sent,
            "output_audio_messages": self._output_audio_messages,
            "output_audio_bytes": self._output_audio_bytes,
            "output_text_chars": self._output_text_chars,
            "first_output_audio_ms": latency_ms(self._first_audio_at_ns),
            "first_output_text_ms": latency_ms(self._first_text_at_ns),
            "sequence_gaps": self._sequence_gaps,
            "dropped_proposals": self._dropped_proposals,
            "last_error": self._last_error,
        }
