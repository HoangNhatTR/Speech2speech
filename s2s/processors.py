"""Pipecat adapters for AudioHub, turn state, policy and playback telemetry."""

from __future__ import annotations

from itertools import cycle

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from s2s.audio_hub import AudioHub
from s2s.events import (
    CommitPolicy,
    Decision,
    FastProposal,
    RiskLevel,
    S2SMode,
    SegmentKind,
    SegmentSource,
    SpeechSegment,
)
from s2s.orchestrator import SessionOrchestrator
from s2s.policy import SegmentPolicy

SAFE_ACKS = (
    "Vâng, tôi đang kiểm tra.",
    "Được, để tôi xem nhé.",
    "Vâng, xin chờ một chút.",
)


class AudioHubProcessor(FrameProcessor):
    def __init__(self, hub: AudioHub, orchestrator: SessionOrchestrator, **kwargs):
        super().__init__(**kwargs)
        self._hub = hub
        self._orchestrator = orchestrator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
            state = self._orchestrator.snapshot()
            self._hub.publish(
                session_id=state.session_id,
                turn_id=state.turn_id,
                revision_id=state.revision_id,
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
        await self.push_frame(frame, direction)


class RealtimeControlProcessor(FrameProcessor):
    """Turns Pipecat frames and S2S proposals into policy-gated TTS frames."""

    def __init__(
        self,
        orchestrator: SessionOrchestrator,
        policy: SegmentPolicy,
        *,
        safe_acks: tuple[str, ...] = SAFE_ACKS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not safe_acks:
            raise ValueError("safe_acks không được rỗng")
        self._orchestrator = orchestrator
        self._policy = policy
        self._ack_cycle = cycle(safe_acks)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._orchestrator.interrupt()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._orchestrator.start_user_turn()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._orchestrator.stop_user_turn()

        await self.push_frame(frame, direction)

        # Static ACK is intentionally only active in ACK_ONLY. SHADOW has zero user
        # impact; SPECULATIVE/PRIMARY wait for a real backend proposal.
        if direction is FrameDirection.DOWNSTREAM and isinstance(frame, UserStoppedSpeakingFrame):
            if self._orchestrator.mode is S2SMode.ACK_ONLY:
                await self.submit_fast_proposal(
                    FastProposal(text=next(self._ack_cycle), kind=SegmentKind.ACK)
                )

    async def submit_fast_proposal(self, proposal: FastProposal) -> None:
        state = self._orchestrator.snapshot()
        segment = SpeechSegment(
            session_id=state.session_id,
            turn_id=proposal.turn_id if proposal.turn_id is not None else state.turn_id,
            revision_id=(
                proposal.revision_id
                if proposal.revision_id is not None
                else state.revision_id
            ),
            generation_id=(
                proposal.generation_id
                if proposal.generation_id is not None
                else state.generation_id
            ),
            source=SegmentSource.FAST,
            kind=proposal.kind,
            text=proposal.text.strip(),
            risk=proposal.risk,
            commit_policy=CommitPolicy.IMMEDIATE,
            claims=proposal.claims,
            requires_tools=proposal.requires_tools,
        )
        self._orchestrator.register_segment(segment)
        decision = self._policy.decide(segment, state)
        self._orchestrator.record_decision(decision)
        if decision.decision is not Decision.ALLOW:
            logger.debug(
                f"[s2s] Không commit fast proposal ({decision.decision.value}): {decision.reason}"
            )
            return
        if not segment.text:
            logger.warning("[s2s] Fast proposal được allow nhưng không có text; bỏ qua")
            return
        # Common Vietnamese renderer for MVP. Raw Mimi audio will get its own output
        # frame adapter only after codec/commit tests pass.
        await self.queue_frame(TTSSpeakFrame(segment.text, append_to_context=False))


class AnchorObservationProcessor(FrameProcessor):
    """Records complete anchor responses as typed segments without delaying TTS."""

    def __init__(self, orchestrator: SessionOrchestrator, policy: SegmentPolicy, **kwargs):
        super().__init__(**kwargs)
        self._orchestrator = orchestrator
        self._policy = policy
        self._parts: list[str] = []
        self._response_state: tuple[int, int, int] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._parts.clear()
                state = self._orchestrator.snapshot()
                self._response_state = (
                    state.turn_id,
                    state.revision_id,
                    state.generation_id,
                )
                self._orchestrator.record_external_event("anchor.started")
            elif isinstance(frame, LLMTextFrame):
                self._parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._commit_observation()
        await self.push_frame(frame, direction)

    def _commit_observation(self) -> None:
        text = "".join(self._parts).strip()
        self._parts.clear()
        response_state = self._response_state
        self._response_state = None
        if not text:
            self._orchestrator.record_external_event("anchor.completed", {"has_text": False})
            return
        state = self._orchestrator.snapshot()
        turn_id, revision_id, generation_id = response_state or (
            state.turn_id,
            state.revision_id,
            state.generation_id,
        )
        segment = SpeechSegment(
            session_id=state.session_id,
            turn_id=turn_id,
            revision_id=revision_id,
            generation_id=generation_id,
            source=SegmentSource.ANCHOR,
            kind=SegmentKind.EXPLANATION,
            text=text,
            risk=RiskLevel.MEDIUM,
            commit_policy=CommitPolicy.ANCHOR_VERIFIED,
        )
        self._orchestrator.register_segment(segment)
        result = self._policy.decide(segment, state)
        self._orchestrator.record_decision(result)
        self._orchestrator.record_external_event(
            "anchor.completed", {"has_text": True, "segment_id": segment.segment_id}
        )


class PlaybackStateProcessor(FrameProcessor):
    """Tracks queued audio and clean playback boundaries for cancellation telemetry."""

    def __init__(self, orchestrator: SessionOrchestrator, **kwargs):
        super().__init__(**kwargs)
        self._orchestrator = orchestrator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._orchestrator.interrupt()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._orchestrator.mark_bot_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._orchestrator.mark_bot_stopped()
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, OutputAudioRawFrame):
            self._orchestrator.queue_audio(frame.num_frames)
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, TTSTextFrame):
            self._orchestrator.record_external_event("tts.segment", {"text_chars": len(frame.text)})
        await self.push_frame(frame, direction)
