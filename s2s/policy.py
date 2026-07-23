"""Deterministic segment risk and commit policy.

No LLM judge is allowed to override stale revisions, unverified tools or sensitive
entities. A learned NLI layer can be inserted later only as an additional soft check.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from s2s.events import (
    CommitDecision,
    CommitPolicy,
    Decision,
    RiskLevel,
    S2SMode,
    SegmentKind,
    SegmentSource,
    SpeechSegment,
)
from s2s.orchestrator import OrchestratorSnapshot

_DIGIT_RE = re.compile(r"\d")
_SENSITIVE_RE = re.compile(
    r"\b(đồng|triệu|tỷ|phần trăm|số tài khoản|mã otp|mã giao dịch|"
    r"đã đặt|đã hủy|thành công|hoàn tất|lãi suất)\b",
    re.IGNORECASE,
)


def contains_sensitive_fact(text: str) -> bool:
    return bool(_DIGIT_RE.search(text) or _SENSITIVE_RE.search(text))


class SegmentPolicy:
    def decide(
        self,
        segment: SpeechSegment,
        snapshot: OrchestratorSnapshot,
        *,
        completed_tools: Collection[str] = (),
    ) -> CommitDecision:
        if segment.session_id != snapshot.session_id:
            return self._result(segment, Decision.DROP, "session_mismatch")
        if segment.turn_id != snapshot.turn_id or segment.revision_id != snapshot.revision_id:
            return self._result(segment, Decision.DROP, "stale_turn_or_revision")

        missing_tools = set(segment.requires_tools) - set(completed_tools)
        if missing_tools:
            return self._result(segment, Decision.WAIT, "tool_not_completed")

        if segment.source in {SegmentSource.ANCHOR, SegmentSource.FALLBACK}:
            if segment.commit_policy is CommitPolicy.TOOL_VERIFIED and not segment.requires_tools:
                return self._result(segment, Decision.WAIT, "tool_dependency_missing")
            return self._result(segment, Decision.ALLOW, "trusted_anchor")

        mode = snapshot.mode
        if mode in {S2SMode.OFF, S2SMode.SHADOW}:
            return self._result(segment, Decision.DROP, f"mode_{mode.value}")

        if segment.claims or contains_sensitive_fact(segment.text):
            return self._result(segment, Decision.WAIT, "fast_factual_content_requires_anchor")
        if segment.risk is RiskLevel.HIGH:
            return self._result(segment, Decision.FALLBACK, "fast_high_risk")
        if segment.commit_policy is not CommitPolicy.IMMEDIATE:
            return self._result(segment, Decision.WAIT, "fast_segment_requires_verification")

        if mode is S2SMode.ACK_ONLY:
            if segment.kind not in {SegmentKind.ACK, SegmentKind.BACKCHANNEL}:
                return self._result(segment, Decision.DROP, "ack_only_kind")
            return self._result(segment, Decision.ALLOW, "safe_ack")

        # SPECULATIVE and PRIMARY still hard-gate facts/tools. PRIMARY means the S2S
        # renderer is preferred, not that it can bypass policy.
        if segment.risk is RiskLevel.MEDIUM or segment.kind in {
            SegmentKind.FACT,
            SegmentKind.CONFIRMATION,
            SegmentKind.TRANSACTION_RESULT,
        }:
            return self._result(segment, Decision.WAIT, "medium_or_factual_requires_anchor")
        return self._result(segment, Decision.ALLOW, "safe_low_risk_fast_segment")

    @staticmethod
    def _result(segment: SpeechSegment, decision: Decision, reason: str) -> CommitDecision:
        return CommitDecision(decision=decision, reason=reason, segment_id=segment.segment_id)
