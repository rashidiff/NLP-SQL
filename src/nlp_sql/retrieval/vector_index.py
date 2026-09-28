"""Vector index interfaces and a deterministic exact in-memory implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nlp_sql.retrieval.embeddings import Vector, cosine_similarity


@dataclass(frozen=True)
class VectorSearchResult:
    key: str
    score: float


@dataclass(frozen=True)
class VectorIndexStats:
    index_size: int
    query_count: int
    selected_search_budget: int


class VectorIndex(Protocol):
    def insert(self, key: str, vector: Vector) -> None: ...

    def delete(self, key: str) -> None: ...

    def search(self, vector: Vector, *, top_k: int, search_budget: int | None = None) -> tuple[VectorSearchResult, ...]: ...

    def persist(self, path: Path) -> None: ...

    def stats(self) -> VectorIndexStats: ...


class InMemoryVectorIndex:
    """Exact vector search implementation behind the VectorIndex contract."""

    def __init__(self) -> None:
        self._vectors: dict[str, Vector] = {}
        self._query_count = 0
        self._selected_search_budget = 0

    def insert(self, key: str, vector: Vector) -> None:
        self._vectors[key] = vector

    def delete(self, key: str) -> None:
        self._vectors.pop(key, None)

    def search(
        self, vector: Vector, *, top_k: int, search_budget: int | None = None
    ) -> tuple[VectorSearchResult, ...]:
        self._query_count += 1
        budget = search_budget or len(self._vectors)
        self._selected_search_budget = budget
        scored = [
            VectorSearchResult(key, cosine_similarity(vector, candidate))
            for key, candidate in self._vectors.items()
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return tuple(scored[:top_k])

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: list(vector) for key, vector in self._vectors.items()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorIndex:
        index = cls()
        if not path.exists():
            return index
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, vector in payload.items():
            index.insert(str(key), tuple(float(value) for value in vector))
        return index

    def stats(self) -> VectorIndexStats:
        return VectorIndexStats(
            index_size=len(self._vectors),
            query_count=self._query_count,
            selected_search_budget=self._selected_search_budget,
        )
