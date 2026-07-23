import unittest

from s2s.audio_hub import AudioHub
from s2s.events import (
    Claim,
    CommitPolicy,
    Decision,
    RiskLevel,
    S2SMode,
    SegmentKind,
    SegmentSource,
    SpeechSegment,
)
from s2s.orchestrator import SessionOrchestrator
from s2s.policy import SegmentPolicy


def make_segment(
    orchestrator: SessionOrchestrator,
    *,
    source: SegmentSource = SegmentSource.FAST,
    kind: SegmentKind = SegmentKind.ACK,
    text: str = "Vâng, tôi đang kiểm tra.",
    risk: RiskLevel = RiskLevel.LOW,
    commit_policy: CommitPolicy = CommitPolicy.IMMEDIATE,
    claims=(),
    requires_tools=(),
) -> SpeechSegment:
    state = orchestrator.snapshot()
    return SpeechSegment(
        session_id=state.session_id,
        turn_id=state.turn_id,
        revision_id=state.revision_id,
        generation_id=state.generation_id,
        source=source,
        kind=kind,
        text=text,
        risk=risk,
        commit_policy=commit_policy,
        claims=tuple(claims),
        requires_tools=tuple(requires_tools),
    )


class SessionOrchestratorTest(unittest.TestCase):
    def test_interruption_cancels_old_generation_and_is_idempotent(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.SHADOW)
        old_token = orchestrator.cancellation_token

        first = orchestrator.interrupt()
        second = orchestrator.start_user_turn()

        self.assertTrue(old_token.cancelled)
        self.assertEqual(old_token.reason, "user_interruption")
        self.assertEqual(first.turn_id, 1)
        self.assertEqual(second.turn_id, 1)
        self.assertEqual(first.generation_id, 1)
        self.assertTrue(second.user_speaking)

    def test_revision_drops_pending_audio_and_rotates_token(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.SPECULATIVE)
        orchestrator.start_user_turn()
        orchestrator.stop_user_turn()
        old_token = orchestrator.cancellation_token
        orchestrator.queue_audio(960)

        state = orchestrator.revise("user_corrected_entity")

        self.assertTrue(old_token.cancelled)
        self.assertEqual(state.revision_id, 2)
        self.assertEqual(state.pending_audio_samples, 0)


class SegmentPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = SegmentPolicy()

    def test_shadow_never_commits_fast_proposal(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.SHADOW)
        segment = make_segment(orchestrator)
        result = self.policy.decide(segment, orchestrator.snapshot())
        self.assertEqual(result.decision, Decision.DROP)

    def test_ack_only_allows_safe_ack_but_not_factual_ack(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.ACK_ONLY)
        safe = make_segment(orchestrator)
        factual = make_segment(orchestrator, text="Đã chuyển thành công 2 triệu đồng.")

        self.assertEqual(
            self.policy.decide(safe, orchestrator.snapshot()).decision,
            Decision.ALLOW,
        )
        self.assertEqual(
            self.policy.decide(factual, orchestrator.snapshot()).decision,
            Decision.WAIT,
        )

    def test_fast_claim_and_medium_fact_require_anchor(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.SPECULATIVE)
        claimed = make_segment(
            orchestrator,
            claims=(Claim(type="amount", value="2000000"),),
        )
        fact = make_segment(
            orchestrator,
            kind=SegmentKind.FACT,
            text="Có vài lựa chọn phù hợp.",
            risk=RiskLevel.MEDIUM,
        )

        self.assertEqual(
            self.policy.decide(claimed, orchestrator.snapshot()).decision,
            Decision.WAIT,
        )
        self.assertEqual(
            self.policy.decide(fact, orchestrator.snapshot()).decision,
            Decision.WAIT,
        )

    def test_anchor_tool_gate_waits_then_allows(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.SHADOW)
        segment = make_segment(
            orchestrator,
            source=SegmentSource.ANCHOR,
            kind=SegmentKind.TRANSACTION_RESULT,
            text="Giao dịch đã hoàn tất.",
            risk=RiskLevel.HIGH,
            commit_policy=CommitPolicy.TOOL_VERIFIED,
            requires_tools=("transfer-17",),
        )

        waiting = self.policy.decide(segment, orchestrator.snapshot())
        allowed = self.policy.decide(
            segment,
            orchestrator.snapshot(),
            completed_tools={"transfer-17"},
        )

        self.assertEqual(waiting.decision, Decision.WAIT)
        self.assertEqual(allowed.decision, Decision.ALLOW)

    def test_stale_revision_is_always_dropped(self):
        orchestrator = SessionOrchestrator("s1", S2SMode.PRIMARY)
        segment = make_segment(orchestrator, source=SegmentSource.ANCHOR)
        orchestrator.revise("new_anchor_revision")

        result = self.policy.decide(segment, orchestrator.snapshot())
        self.assertEqual(result.decision, Decision.DROP)
        self.assertEqual(result.reason, "stale_turn_or_revision")


class AudioHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_ring_is_bounded_and_slow_subscriber_drops_oldest(self):
        hub = AudioHub(max_buffer_ms=40, subscriber_queue_frames=1)
        subscription = hub.subscribe("slow-model")
        pcm_20ms = b"\x00\x00" * 320  # 20 ms, mono PCM16 at 16 kHz

        for _ in range(3):
            hub.publish(
                session_id="s1",
                turn_id=1,
                revision_id=1,
                audio=pcm_20ms,
                sample_rate=16_000,
                num_channels=1,
            )

        stats = hub.stats()
        newest = await subscription.get()

        self.assertLessEqual(stats.buffered_ms, 40.001)
        self.assertEqual(len(hub.buffered_chunks()), 2)
        self.assertEqual(stats.dropped_subscriber_frames, 2)
        self.assertEqual(newest.sequence, 2)
        hub.close()

    async def test_close_wakes_subscriber(self):
        hub = AudioHub(max_buffer_ms=100, subscriber_queue_frames=2)
        subscription = hub.subscribe("model")
        hub.close()
        self.assertIsNone(await subscription.get())


if __name__ == "__main__":
    unittest.main()
