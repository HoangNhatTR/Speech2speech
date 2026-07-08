"""Interface chung cho các service đứng sau Runtime Scheduler (runtime/dispatcher.py).

Chỉ 7/9 service trong sơ đồ dùng interface này: Speech chạy trong-tiến-trình ở Gateway
(lý do latency audio), Monitoring là subscriber thụ động, không nhận request/response.
Xem docs/platform-architecture.md.
"""

from abc import ABC, abstractmethod


class BaseService(ABC):
    @abstractmethod
    async def handle(self, request: dict) -> dict:
        """Xử lý một request và trả về response. Cả hai đều là dict JSON-hoá được."""
