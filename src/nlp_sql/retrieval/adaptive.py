"""Adaptive vector retrieval policy inspired by workload-shift-aware indexing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from nlp_sql.retrieval.embeddings import Vector
from nlp_sql.retrieval.telemetry import RetrievalEvent, RetrievalTelemetry, Timer
from nlp_sql.retrieval.vector_index import VectorIndex, VectorIndexStats, VectorSearchResult


@dataclass(frozen=True)
class AdaptivePolicy:
    easy_margin: float = 0.20
    shift_threshold: float = 0.45
    min_budget: int = 8
    default_budget: int = 16
    max_budget: int = 64
    recent_window: int = 50


@dataclass
class AdaptiveVectorIndex:
    index: VectorIndex
    policy: AdaptivePolicy = AdaptivePolicy()
    telemetry: RetrievalTelemetry = field(default_factory=RetrievalTelemetry)
    _historical_hits: Counter[str] = field(default_factory=Counter)
    _recent_hits: deque[str] = field(default_factory=deque)
    _last_budget: int = 16
    _workload_shifted: bool = False

    def insert(self, key: str, vector: Vector) -> None:
        self.index.insert(key, vector)

    def delete(self, key: str) -> None:
        self.index.delete(key)

    def search(self, query: str, vector: Vector, *, top_k: int) -> tuple[VectorSearchResult, ...]:
        budget = self._choose_budget(top_k)
        with Timer() as timer:
            results = self.index.search(vector, top_k=top_k, search_budget=budget)
        margin = self._margin(results)
        if margin < self.policy.easy_margin and budget < self.policy.max_budget:
            budget = self.policy.max_budget
            with Timer() as timer:
                results = self.index.search(vector, top_k=top_k, search_budget=budget)
            margin = self._margin(results)
        selected = tuple(result.key for result in results)
        self._record_hits(selected)
        self._last_budget = budget
        self.telemetry.record(
            RetrievalEvent(
                query=query,
                latency_ms=timer.elapsed_ms,
                selected_keys=selected,
                margin=margin,
                search_budget=budget,
            )
        )
        return results

    def stats(self) -> dict[str, object]:
        base_stats: VectorIndexStats = self.index.stats()
        return {
            "index_size": base_stats.index_size,
            "query_count": base_stats.query_count,
            "selected_search_budget": self._last_budget,
            "workload_shifted": self._workload_shifted,
            "telemetry": self.telemetry.snapshot(),
        }

    def _choose_budget(self, top_k: int) -> int:
        if self._workload_shifted:
            return self.policy.max_budget
        return max(self.policy.min_budget, min(self.policy.default_budget, top_k * 4))

    def _record_hits(self, keys: tuple[str, ...]) -> None:
        for key in keys:
            self._historical_hits[key] += 1
            self._recent_hits.append(key)
            while len(self._recent_hits) > self.policy.recent_window:
                self._recent_hits.popleft()
        self._workload_shifted = self._detect_shift()

    def _detect_shift(self) -> bool:
        if len(self._recent_hits) < self.policy.recent_window:
            return False
        recent = Counter(self._recent_hits)
        recent_total = sum(recent.values())
        historical_total = sum(self._historical_hits.values())
        if recent_total == 0 or historical_total == 0:
            return False
        keys = set(recent) | set(self._historical_hits)
        distance = 0.0
        for key in keys:
            recent_share = recent[key] / recent_total
            historical_share = self._historical_hits[key] / historical_total
            distance += abs(recent_share - historical_share)
        return distance / 2 >= self.policy.shift_threshold

    def _margin(self, results: tuple[VectorSearchResult, ...]) -> float:
        if len(results) < 2:
            return 1.0 if results else 0.0
        return results[0].score - results[1].score
