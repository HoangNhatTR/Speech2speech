"""Pluggable audio-token S2S backend contract and a zero-cost shadow probe."""

from __future__ import annotations

from abc import ABC, abstractmethod

from s2s.events import AudioChunk, FastProposal
from s2s.orchestrator import OrchestratorSnapshot


class AudioTokenS2SBackend(ABC):
    """Contract for a future Mimi/Moshi implementation.

    Implementations must be causal and must not block the media producer. They receive
    ordered PCM16 chunks from ``AudioHub`` and may return zero or more proposals. The
    Session Orchestrator and Segment Policy remain authoritative.
    """

    name = "abstract"

    async def start(self, session_id: str) -> None:
        pass

    @abstractmethod
    async def process_audio(
        self, chunk: AudioChunk, snapshot: OrchestratorSnapshot
    ) -> list[FastProposal]:
        raise NotImplementedError

    async def stop(self) -> None:
        pass

    def stats(self) -> dict[str, int | float | str | bool]:
        return {"backend": self.name}


class ProbeS2SBackend(AudioTokenS2SBackend):
    """Validates realtime fan-out without pretending to be a speech model."""

    name = "probe"

    def __init__(self) -> None:
        self.frames = 0
        self.audio_ms = 0.0
        self.last_sequence = -1

    async def process_audio(
        self, chunk: AudioChunk, snapshot: OrchestratorSnapshot
    ) -> list[FastProposal]:
        self.frames += 1
        self.audio_ms += chunk.duration_ms
        self.last_sequence = chunk.sequence
        return []

    def stats(self) -> dict[str, int | float | str]:
        return {
            "backend": self.name,
            "frames": self.frames,
            "audio_ms": round(self.audio_ms, 3),
            "last_sequence": self.last_sequence,
        }


def build_s2s_backend(name: str) -> AudioTokenS2SBackend:
    normalized = name.strip().lower()
    if normalized == "probe":
        return ProbeS2SBackend()
    if normalized in {"moshi", "moshi_ws"}:
        # Lazy import keeps sphn/Opus and model-side dependencies out of startup when
        # the safe default probe is selected.
        from s2s.moshi_backend import MoshiWebSocketBackend

        return MoshiWebSocketBackend.from_env()
    raise ValueError(
        f"S2S_SHADOW_BACKEND={name!r} chưa được hỗ trợ. "
        "Chọn 'probe' hoặc 'moshi_ws'."
    )
