"""Background consumer that isolates experimental S2S failures from anchor audio."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from s2s.audio_hub import AudioSubscription
from s2s.backend import AudioTokenS2SBackend
from s2s.events import FastProposal
from s2s.orchestrator import SessionOrchestrator

ProposalHandler = Callable[[FastProposal], Awaitable[None]]


class ShadowS2SRunner:
    def __init__(
        self,
        *,
        subscription: AudioSubscription,
        backend: AudioTokenS2SBackend,
        orchestrator: SessionOrchestrator,
        on_proposal: ProposalHandler,
        telemetry_every_frames: int = 100,
    ):
        self._subscription = subscription
        self._backend = backend
        self._orchestrator = orchestrator
        self._on_proposal = on_proposal
        self._telemetry_every_frames = max(1, telemetry_every_frames)
        self._task: asyncio.Task | None = None
        self._stopped = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name=f"s2s-shadow-{self._orchestrator.session_id}")

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._subscription.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        else:
            await self._backend.stop()

    async def _run(self) -> None:
        processed = 0
        try:
            # Connect/load outside the WebRTC callback. A missing or busy experimental
            # model must never delay the first Anchor greeting.
            try:
                await self._backend.start(self._orchestrator.session_id)
                self._orchestrator.record_external_event(
                    "shadow.started", {"backend": self._backend.name}
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[s2s] Không khởi động được shadow backend; anchor tiếp tục: {exc}")
                self._orchestrator.record_external_event(
                    "shadow.start_error",
                    {"backend": self._backend.name, "error": str(exc)},
                )
                return

            while True:
                chunk = await self._subscription.get()
                if chunk is None:
                    break
                try:
                    proposals = await self._backend.process_audio(
                        chunk, self._orchestrator.snapshot()
                    )
                    for proposal in proposals:
                        await self._on_proposal(proposal)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Shadow failure is observable but never allowed to block/cancel anchor.
                    logger.exception(f"[s2s] shadow backend lỗi, anchor tiếp tục: {exc}")
                    self._orchestrator.record_external_event(
                        "shadow.error", {"backend": self._backend.name, "error": str(exc)}
                    )
                processed += 1
                if processed % self._telemetry_every_frames == 0:
                    self._orchestrator.record_external_event("shadow.telemetry", self._backend.stats())
        finally:
            await self._backend.stop()
