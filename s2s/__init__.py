"""Realtime dual-path Speech-to-Speech control plane.

The package deliberately contains no model weights and never sends audio through
NATS/Ray. It provides the stable contracts that a local Mimi/Moshi sidecar can plug
into while the verified ASR -> LLM -> TTS path remains the production anchor.
"""

from s2s.events import (
    CommitDecision,
    CommitPolicy,
    Decision,
    FastProposal,
    RiskLevel,
    S2SMode,
    SegmentKind,
    SegmentSource,
    SpeechSegment,
)
from s2s.orchestrator import SessionOrchestrator, orchestrator_registry

__all__ = [
    "CommitDecision",
    "CommitPolicy",
    "Decision",
    "FastProposal",
    "RiskLevel",
    "S2SMode",
    "SegmentKind",
    "SegmentSource",
    "SessionOrchestrator",
    "SpeechSegment",
    "orchestrator_registry",
]
