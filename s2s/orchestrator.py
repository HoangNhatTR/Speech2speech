"""Authoritative per-session turn, revision, cancellation and commit state."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from s2s.events import CommitDecision, Decision, S2SMode, SpeechSegment


class CancellationToken:
    """Generation-scoped cooperative cancellation token."""

    def __init__(self, generation_id: int):
        self.generation_id = generation_id
        self._event = asyncio.Event()
        self.reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def cancel(self, reason: str) -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self.reason or "generation cancelled")


@dataclass(frozen=True, slots=True)
class OrchestratorSnapshot:
    session_id: str
    mode: S2SMode
    turn_id: int
    revision_id: int
    generation_id: int
    user_speaking: bool
    bot_speaking: bool
    pending_audio_samples: int
    played_audio_samples: int
    generated_segments: int
    committed_segments: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data


@dataclass(frozen=True, slots=True)
class StateEvent:
    name: str
    turn_id: int
    revision_id: int
    generation_id: int
    details: dict[str, Any] = field(default_factory=dict)
    at_ns: int = field(default_factory=time.monotonic_ns)


class SessionOrchestrator:
    """Single source of truth for one realtime voice session.

    Mutations are synchronous by design: a Pipecat session executes them on one event
    loop, making state transitions atomic without inserting awaits into the media path.
    External model work uses the generation ``CancellationToken``.
    """

    def __init__(self, session_id: str, mode: S2SMode, *, event_history: int = 256):
        self.session_id = session_id
        self.mode = mode
        self.turn_id = 0
        self.revision_id = 0
        self.generation_id = 0
        self.user_speaking = False
        self.bot_speaking = False
        self.pending_audio_samples = 0
        self.played_audio_samples = 0
        self._segments: dict[str, SpeechSegment] = {}
        self._decisions: dict[str, CommitDecision] = {}
        self._committed: set[str] = set()
        self._events: deque[StateEvent] = deque(maxlen=event_history)
        self._token = CancellationToken(self.generation_id)
        self._closed = False
        self._record("session.created", {"mode": mode.value})

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._token

    def snapshot(self) -> OrchestratorSnapshot:
        return OrchestratorSnapshot(
            session_id=self.session_id,
            mode=self.mode,
            turn_id=self.turn_id,
            revision_id=self.revision_id,
            generation_id=self.generation_id,
            user_speaking=self.user_speaking,
            bot_speaking=self.bot_speaking,
            pending_audio_samples=self.pending_audio_samples,
            played_audio_samples=self.played_audio_samples,
            generated_segments=len(self._segments),
            committed_segments=len(self._committed),
        )

    def recent_events(self) -> list[StateEvent]:
        return list(self._events)

    def start_user_turn(self, reason: str = "user_started") -> OrchestratorSnapshot:
        self._ensure_open()
        if self.user_speaking:
            return self.snapshot()
        self._rotate_generation(reason)
        self.turn_id += 1
        self.revision_id += 1
        self.user_speaking = True
        self.bot_speaking = False
        self.pending_audio_samples = 0
        self._record("turn.started", {"reason": reason})
        return self.snapshot()

    def stop_user_turn(self) -> OrchestratorSnapshot:
        self._ensure_open()
        if self.user_speaking:
            self.user_speaking = False
            self._record("turn.user_stopped")
        return self.snapshot()

    def interrupt(self, reason: str = "user_interruption") -> OrchestratorSnapshot:
        """Cancel stale work and atomically open the interrupting user turn.

        Pipecat may emit both ``InterruptionFrame`` and ``UserStartedSpeakingFrame``.
        ``user_speaking`` makes the second transition idempotent.
        """

        self._ensure_open()
        if self.user_speaking:
            return self.snapshot()
        self._rotate_generation(reason)
        self.turn_id += 1
        self.revision_id += 1
        self.user_speaking = True
        self.bot_speaking = False
        dropped = self.pending_audio_samples
        self.pending_audio_samples = 0
        self._record("turn.interrupted", {"reason": reason, "dropped_audio_samples": dropped})
        return self.snapshot()

    def revise(self, reason: str) -> OrchestratorSnapshot:
        self._ensure_open()
        self._rotate_generation(reason)
        self.revision_id += 1
        self.pending_audio_samples = 0
        self._record("turn.revised", {"reason": reason})
        return self.snapshot()

    def register_segment(self, segment: SpeechSegment) -> None:
        self._ensure_open()
        if segment.session_id != self.session_id:
            raise ValueError("segment thuộc session khác")
        self._segments[segment.segment_id] = segment
        self._record(
            "segment.generated",
            {"segment_id": segment.segment_id, "source": segment.source.value, "kind": segment.kind.value},
        )

    def record_decision(self, result: CommitDecision) -> None:
        self._ensure_open()
        if result.segment_id not in self._segments:
            raise KeyError(f"segment chưa đăng ký: {result.segment_id}")
        self._decisions[result.segment_id] = result
        if result.decision is Decision.ALLOW:
            self._committed.add(result.segment_id)
        self._record(
            "segment.decision",
            {"segment_id": result.segment_id, "decision": result.decision.value, "reason": result.reason},
        )

    def queue_audio(self, num_samples: int) -> None:
        self._ensure_open()
        if num_samples > 0:
            self.pending_audio_samples += num_samples

    def mark_bot_started(self) -> None:
        self._ensure_open()
        self.bot_speaking = True
        self._record("playback.started")

    def mark_bot_stopped(self) -> None:
        self._ensure_open()
        self.bot_speaking = False
        # Pipecat 0.0.108 does not expose a sample-accurate playback ACK. At a clean
        # BotStopped boundary, queued audio is the best conservative completion marker.
        self.played_audio_samples += self.pending_audio_samples
        self.pending_audio_samples = 0
        self._record("playback.stopped")

    def record_external_event(self, name: str, details: dict[str, Any] | None = None) -> None:
        self._ensure_open()
        self._record(name, details or {})

    def close(self) -> None:
        if self._closed:
            return
        self._token.cancel("session_closed")
        self.pending_audio_samples = 0
        self._record("session.closed")
        self._closed = True

    def _rotate_generation(self, reason: str) -> None:
        self._token.cancel(reason)
        self.generation_id += 1
        self._token = CancellationToken(self.generation_id)

    def _record(self, name: str, details: dict[str, Any] | None = None) -> None:
        self._events.append(
            StateEvent(
                name=name,
                turn_id=self.turn_id,
                revision_id=self.revision_id,
                generation_id=self.generation_id,
                details=details or {},
            )
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("session orchestrator đã đóng")


class OrchestratorRegistry:
    """Process-local registry used by the dashboard and session cleanup."""

    def __init__(self) -> None:
        self._items: dict[str, SessionOrchestrator] = {}

    def create(self, session_id: str, mode: S2SMode) -> SessionOrchestrator:
        if session_id in self._items:
            raise KeyError(f"orchestrator đã tồn tại: {session_id}")
        orchestrator = SessionOrchestrator(session_id, mode)
        self._items[session_id] = orchestrator
        return orchestrator

    def get(self, session_id: str) -> SessionOrchestrator | None:
        return self._items.get(session_id)

    def remove(self, session_id: str) -> None:
        orchestrator = self._items.pop(session_id, None)
        if orchestrator:
            orchestrator.close()

    def snapshots(self) -> list[dict[str, Any]]:
        return [item.snapshot().to_dict() for item in self._items.values()]


orchestrator_registry = OrchestratorRegistry()
