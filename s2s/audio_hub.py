"""Bounded in-process audio fan-out for anchor and speech-native consumers."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass

from s2s.events import AudioChunk


@dataclass(frozen=True, slots=True)
class AudioHubStats:
    published_frames: int
    dropped_subscriber_frames: int
    buffered_ms: float
    subscribers: int


class AudioSubscription:
    def __init__(self, hub: "AudioHub", name: str, queue: asyncio.Queue[AudioChunk | None]):
        self._hub = hub
        self.name = name
        self._queue = queue
        self._closed = False

    async def get(self) -> AudioChunk | None:
        return await self._queue.get()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub.unsubscribe(self.name)


class AudioHub:
    """A session-local ring buffer with bounded subscriber queues.

    Slow experimental consumers drop their oldest frame instead of back-pressuring
    WebRTC. This is essential: the shadow model may fail without delaying anchor ASR.
    """

    def __init__(self, *, max_buffer_ms: int = 30_000, subscriber_queue_frames: int = 128):
        if max_buffer_ms <= 0 or subscriber_queue_frames <= 0:
            raise ValueError("AudioHub limits phải lớn hơn 0")
        self.max_buffer_ms = max_buffer_ms
        self.subscriber_queue_frames = subscriber_queue_frames
        self._buffer: deque[AudioChunk] = deque()
        self._buffered_ms = 0.0
        self._subscribers: dict[str, asyncio.Queue[AudioChunk | None]] = {}
        self._sequence = defaultdict(int)
        self._published_frames = 0
        self._dropped_subscriber_frames = 0
        self._closed = False

    def subscribe(self, name: str) -> AudioSubscription:
        if self._closed:
            raise RuntimeError("AudioHub đã đóng")
        if name in self._subscribers:
            raise KeyError(f"audio subscriber đã tồn tại: {name}")
        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(
            maxsize=self.subscriber_queue_frames
        )
        self._subscribers[name] = queue
        return AudioSubscription(self, name, queue)

    def unsubscribe(self, name: str) -> None:
        queue = self._subscribers.pop(name, None)
        if queue is not None:
            self._signal_closed(queue)

    def publish(
        self,
        *,
        session_id: str,
        turn_id: int,
        revision_id: int,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> AudioChunk:
        if self._closed:
            raise RuntimeError("AudioHub đã đóng")
        if sample_rate <= 0 or num_channels <= 0:
            raise ValueError("sample_rate và num_channels phải lớn hơn 0")

        sequence = self._sequence[session_id]
        self._sequence[session_id] += 1
        chunk = AudioChunk(
            session_id=session_id,
            turn_id=turn_id,
            revision_id=revision_id,
            sequence=sequence,
            audio=audio,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._buffer.append(chunk)
        self._buffered_ms += chunk.duration_ms
        self._published_frames += 1
        self._trim_buffer()

        for queue in self._subscribers.values():
            if queue.full():
                try:
                    queue.get_nowait()
                    self._dropped_subscriber_frames += 1
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(chunk)
        return chunk

    def buffered_chunks(self) -> tuple[AudioChunk, ...]:
        return tuple(self._buffer)

    def stats(self) -> AudioHubStats:
        return AudioHubStats(
            published_frames=self._published_frames,
            dropped_subscriber_frames=self._dropped_subscriber_frames,
            buffered_ms=round(self._buffered_ms, 3),
            subscribers=len(self._subscribers),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in self._subscribers.values():
            self._signal_closed(queue)
        self._subscribers.clear()
        self._buffer.clear()
        self._buffered_ms = 0.0

    def _trim_buffer(self) -> None:
        while self._buffer and self._buffered_ms > self.max_buffer_ms:
            removed = self._buffer.popleft()
            self._buffered_ms = max(0.0, self._buffered_ms - removed.duration_ms)

    @staticmethod
    def _signal_closed(queue: asyncio.Queue[AudioChunk | None]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(None)
