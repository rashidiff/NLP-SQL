"""Telemetry for schema retrieval and adaptive vector search."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalEvent:
    query: str
    latency_ms: float
    selected_keys: tuple[str, ...]
    margin: float
    search_budget: int


@dataclass
class RetrievalTelemetry:
    max_recent_events: int = 100
    query_count: int = 0
    total_latency_ms: float = 0.0
    access_frequency: Counter[str] = field(default_factory=Counter)
    recent_events: deque[RetrievalEvent] = field(default_factory=deque)

    def record(self, event: RetrievalEvent) -> None:
        self.query_count += 1
        self.total_latency_ms += event.latency_ms
        self.access_frequency.update(event.selected_keys)
        self.recent_events.append(event)
        while len(self.recent_events) > self.max_recent_events:
            self.recent_events.popleft()

    def snapshot(self) -> dict[str, object]:
        average_latency = self.total_latency_ms / self.query_count if self.query_count else 0.0
        return {
            "query_count": self.query_count,
            "average_latency_ms": average_latency,
            "access_frequency": dict(self.access_frequency),
            "recent_event_count": len(self.recent_events),
        }


class Timer:
    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._started) * 1000

    elapsed_ms: float = 0.0
