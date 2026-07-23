import asyncio
import unittest

import numpy as np
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext

from duplex.interruption_recovery import InterruptionRecoveryProcessor
from duplex.turn_strategies import VietnameseBackchannelTurnStartStrategy
from selfhost.tts_server import SynthesizeRequest, _float_audio_to_pcm16, _stream_pcm16


class FakeStreamingTTS:
    sample_rate = 48000

    def __init__(self):
        self.calls = []

    def infer_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        yield np.array([-1.0, 0.0], dtype=np.float32)
        yield np.array([0.5, 1.0], dtype=np.float32)


class StreamingTTSTest(unittest.TestCase):
    def test_pcm16_conversion_clips_and_preserves_chunks(self):
        pcm = _float_audio_to_pcm16(np.array([-2.0, -1.0, 0.0, 0.5, 2.0]))
        samples = np.frombuffer(pcm, dtype="<i2")
        np.testing.assert_array_equal(samples, [-32767, -32767, 0, 16383, 32767])

    def test_stream_pcm_yields_before_full_result(self):
        tts = FakeStreamingTTS()
        chunks = _stream_pcm16(tts, SynthesizeRequest(text="Xin chào"))

        first = next(chunks)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(list(chunks)), 1)
        self.assertEqual(tts.calls[0][0], "Xin chào")


class DelayedVADBargeInTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.strategy = VietnameseBackchannelTurnStartStrategy(barge_in_delay_ms=30)
        self.started = 0

        @self.strategy.event_handler("on_user_turn_started")
        async def _on_started(_strategy, _params):
            self.started += 1

        await self.strategy.process_frame(BotStartedSpeakingFrame())

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def test_vad_interrupts_local_backend_without_interim_transcript(self):
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.06)
        self.assertEqual(self.started, 1)

    async def test_backchannel_cancels_pending_vad_interruption(self):
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await self.strategy.process_frame(
            InterimTranscriptionFrame(text="dạ", user_id="user", timestamp="0")
        )
        await asyncio.sleep(0.06)
        self.assertEqual(self.started, 0)

    async def test_real_interruption_does_not_wait_for_vad_timeout(self):
        await self.strategy.process_frame(VADUserStartedSpeakingFrame())
        await self.strategy.process_frame(
            InterimTranscriptionFrame(text="khoan đã", user_id="user", timestamp="0")
        )
        self.assertEqual(self.started, 1)


class InterruptionRecoveryTest(unittest.TestCase):
    def test_completed_turn_is_not_reported_in_later_interruption(self):
        context = LLMContext(messages=[])
        processor = InterruptionRecoveryProcessor(context)

        processor._track_frame(TTSTextFrame("lượt cũ", aggregated_by="sentence"))
        processor._track_frame(BotStoppedSpeakingFrame())
        processor._track_frame(InterruptionFrame())
        self.assertEqual(context.messages, [])

        processor._track_frame(TTSTextFrame("lượt hiện tại", aggregated_by="sentence"))
        processor._track_frame(InterruptionFrame())
        self.assertEqual(len(context.messages), 1)
        self.assertIn("lượt hiện tại", context.messages[0]["content"])
        self.assertNotIn("lượt cũ", context.messages[0]["content"])
