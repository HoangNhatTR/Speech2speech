"""Typed contracts shared by the fast S2S path, anchor and verifier."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class S2SMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ACK_ONLY = "ack_only"
    SPECULATIVE = "speculative"
    PRIMARY = "primary"

    @classmethod
    def parse(cls, value: str | None) -> "S2SMode":
        normalized = (value or cls.SHADOW.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"S2S_MODE={value!r} không hợp lệ; chọn một trong: {choices}") from exc


class SegmentSource(str, Enum):
    FAST = "fast"
    ANCHOR = "anchor"
    FALLBACK = "fallback"


class SegmentKind(str, Enum):
    ACK = "ack"
    BACKCHANNEL = "backchannel"
    SOCIAL = "social"
    EXPLANATION = "explanation"
    FACT = "fact"
    CONFIRMATION = "confirmation"
    TRANSACTION_RESULT = "transaction_result"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommitPolicy(str, Enum):
    IMMEDIATE = "immediate"
    ANCHOR_VERIFIED = "anchor_verified"
    TOOL_VERIFIED = "tool_verified"


class Decision(str, Enum):
    ALLOW = "allow"
    WAIT = "wait"
    DROP = "drop"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class Claim:
    type: str
    value: str
    source_id: str | None = None
    verified: bool = False


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    session_id: str
    turn_id: int
    revision_id: int
    generation_id: int
    source: SegmentSource
    kind: SegmentKind
    text: str
    risk: RiskLevel = RiskLevel.LOW
    commit_policy: CommitPolicy = CommitPolicy.IMMEDIATE
    claims: tuple[Claim, ...] = ()
    requires_tools: tuple[str, ...] = ()
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at_ns: int = field(default_factory=time.monotonic_ns)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("source", "kind", "risk", "commit_policy"):
            data[key] = getattr(self, key).value
        return data


@dataclass(frozen=True, slots=True)
class FastProposal:
    """Model-neutral output from a speech-native backend.

    The MVP renders ``text`` through the common Vietnamese TTS. ``audio`` is reserved
    for a future Mimi decoder path and is never committed without the same policy gate.
    """

    text: str
    kind: SegmentKind
    risk: RiskLevel = RiskLevel.LOW
    claims: tuple[Claim, ...] = ()
    requires_tools: tuple[str, ...] = ()
    audio: bytes | None = None
    sample_rate: int | None = None
    # Provenance is optional for simple deterministic backchannels. A model running
    # asynchronously must attach it so a late prefix cannot be relabelled as belonging
    # to a newer user turn/revision.
    turn_id: int | None = None
    revision_id: int | None = None
    generation_id: int | None = None


@dataclass(frozen=True, slots=True)
class CommitDecision:
    decision: Decision
    reason: str
    segment_id: str
    decided_at_ns: int = field(default_factory=time.monotonic_ns)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


@dataclass(frozen=True, slots=True)
class AudioChunk:
    session_id: str
    turn_id: int
    revision_id: int
    sequence: int
    audio: bytes
    sample_rate: int
    num_channels: int
    captured_at_ns: int = field(default_factory=time.monotonic_ns)

    @property
    def num_frames(self) -> int:
        width = self.num_channels * 2  # PCM16
        return len(self.audio) // width if width else 0

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_frames * 1000 / self.sample_rate
