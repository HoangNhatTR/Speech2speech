from __future__ import annotations

import asyncio
import unittest

import numpy as np
from aiohttp import WSMsgType, web

from s2s.audio_hub import AudioHub
from s2s.backend import AudioTokenS2SBackend
from s2s.events import AudioChunk, S2SMode
from s2s.moshi_backend import MOSHI_SAMPLE_RATE, MoshiWebSocketBackend
from s2s.orchestrator import OrchestratorSnapshot, SessionOrchestrator
from s2s.shadow import ShadowS2SRunner


def snapshot() -> OrchestratorSnapshot:
    return OrchestratorSnapshot(
        session_id="session-1",
        mode=S2SMode.SHADOW,
        turn_id=7,
        revision_id=3,
        generation_id=9,
        user_speaking=True,
        bot_speaking=False,
        pending_audio_samples=0,
        played_audio_samples=0,
        generated_segments=0,
        committed_segments=0,
    )


def pcm_chunk(sequence: int, *, sample_rate: int = 16_000) -> AudioChunk:
    # 20 ms sine-like PCM is enough for OpusStreamWriter to produce packets after a
    # few calls, while keeping the protocol integration test fast.
    frames = sample_rate // 50
    values = (np.arange(frames, dtype=np.int16) * 31).astype("<i2")
    return AudioChunk(
        session_id="session-1",
        turn_id=7,
        revision_id=3,
        sequence=sequence,
        audio=values.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
    )


class MoshiBackendUnitTest(unittest.TestCase):
    def test_remote_audio_endpoint_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            MoshiWebSocketBackend(url="wss://example.com/api/chat")
        backend = MoshiWebSocketBackend(
            url="wss://example.com/api/chat", allow_remote=True
        )
        self.assertEqual(backend.url, "wss://example.com/api/chat")

    def test_text_proposal_keeps_original_turn_provenance(self):
        backend = MoshiWebSocketBackend()
        backend._last_snapshot = snapshot()
        backend._observe_text("A model prefix.")

        proposals = backend._drain_proposals()
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.turn_id, 7)
        self.assertEqual(proposal.revision_id, 3)
        self.assertEqual(proposal.generation_id, 9)

    def test_pcm_stereo_is_mixed_to_float_mono(self):
        backend = MoshiWebSocketBackend()
        stereo = np.array([[32767, -32768], [16384, 16384]], dtype="<i2")
        chunk = AudioChunk(
            session_id="s",
            turn_id=0,
            revision_id=0,
            sequence=0,
            audio=stereo.tobytes(),
            sample_rate=MOSHI_SAMPLE_RATE,
            num_channels=2,
        )
        mono = backend._to_mono_float32(chunk)
        self.assertEqual(mono.shape, (2,))
        self.assertAlmostEqual(float(mono[0]), -0.5 / 32768.0, places=6)
        self.assertAlmostEqual(float(mono[1]), 0.5, places=6)


class MoshiProtocolIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received: list[bytes] = []
        app = web.Application()

        async def chat(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_bytes(b"\x00")
            async for message in ws:
                if message.type is WSMsgType.BINARY:
                    self.received.append(message.data)
                    await ws.send_bytes(b"\x02shadow prefix.")
                    await ws.send_bytes(b"\x01mock-opus-output")
            return ws

        app.router.add_get("/api/chat", chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets
        self.port = sockets[0].getsockname()[1]
        self.backend = MoshiWebSocketBackend(
            url=f"ws://127.0.0.1:{self.port}/api/chat", connect_timeout_s=1
        )

    async def asyncTearDown(self):
        await self.backend.stop()
        await self.runner.cleanup()

    async def test_official_handshake_audio_and_observation_protocol(self):
        await self.backend.start("session-1")
        proposals = []
        for sequence in range(12):
            proposals.extend(
                await self.backend.process_audio(pcm_chunk(sequence), snapshot())
            )
        await asyncio.sleep(0.05)
        proposals.extend(
            await self.backend.process_audio(pcm_chunk(12), snapshot())
        )

        self.assertTrue(any(message.startswith(b"\x01") for message in self.received))
        self.assertTrue(proposals)
        self.assertEqual(proposals[0].revision_id, 3)
        stats = self.backend.stats()
        self.assertGreater(stats["output_audio_messages"], 0)
        self.assertGreater(stats["output_text_chars"], 0)
        self.assertEqual(stats["last_error"], "")


class FailingBackend(AudioTokenS2SBackend):
    name = "failing"

    async def start(self, session_id: str) -> None:
        raise RuntimeError("model unavailable")

    async def process_audio(self, chunk, state):
        raise AssertionError("must not process after failed start")


class ShadowIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_failure_does_not_escape_to_anchor_callback(self):
        hub = AudioHub(max_buffer_ms=100, subscriber_queue_frames=2)
        orchestrator = SessionOrchestrator("session-1", S2SMode.SHADOW)
        runner = ShadowS2SRunner(
            subscription=hub.subscribe("failing"),
            backend=FailingBackend(),
            orchestrator=orchestrator,
            on_proposal=lambda proposal: asyncio.sleep(0),
        )

        await runner.start()  # Must return before model connection/loading completes.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        names = [event.name for event in orchestrator.recent_events()]
        self.assertIn("shadow.start_error", names)
        await runner.stop()
        hub.close()


if __name__ == "__main__":
    unittest.main()
