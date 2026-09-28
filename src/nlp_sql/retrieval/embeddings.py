"""Local embedding backends and disk-backed embedding storage."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nlp_sql.retrieval.base import stable_hash, tokenize


Vector = tuple[float, ...]


class EmbeddingBackend(Protocol):
    model_id: str
    dimensions: int

    def embed(self, texts: tuple[str, ...]) -> tuple[Vector, ...]:
        """Embed texts locally without calling external APIs."""


@dataclass(frozen=True)
class LocalHashEmbeddingBackend:
    """Small deterministic local embedding backend based on hashed token n-grams.

    This is intentionally simple and replaceable. It gives the retrieval layer a
    local semantic-ish signal without requiring an external model or API.
    """

    dimensions: int = 128
    model_id: str = "local-hash-ngram-v1"

    def embed(self, texts: tuple[str, ...]) -> tuple[Vector, ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> Vector:
        values = [0.0 for _ in range(self.dimensions)]
        tokens = tokenize(text)
        grams = list(tokens)
        grams.extend(" ".join(tokens[index : index + 2]) for index in range(len(tokens) - 1))
        for gram in grams:
            digest = stable_hash(gram)
            bucket = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            values[bucket] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return tuple(values)
        return tuple(value / norm for value in values)


@dataclass(frozen=True)
class SentenceTransformerEmbeddingBackend:
    """Optional local sentence-transformers backend.

    The dependency is loaded lazily so the core package continues to work
    without installing large optional libraries.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def dimensions(self) -> int:
        probe = self.embed(("dimension probe",))[0]
        return len(probe)

    def embed(self, texts: tuple[str, ...]) -> tuple[Vector, ...]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer(self.model_name)
        encoded = model.encode(list(texts), normalize_embeddings=True)
        return tuple(tuple(float(value) for value in vector) for vector in encoded)


@dataclass(frozen=True)
class EmbeddingRecord:
    key: str
    vector: Vector
    dimensions: int
    model_id: str
    source_hash: str
    metadata: dict[str, str]


class EmbeddingStore:
    """SQLite-backed embedding key-value store with bounded LRU hot cache."""

    def __init__(self, path: Path, *, max_cache_items: int = 256) -> None:
        self._path = path
        self._max_cache_items = max_cache_items
        self._cache: OrderedDict[str, EmbeddingRecord] = OrderedDict()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_many(
        self, keys: tuple[str, ...], *, model_id: str, source_hashes: dict[str, str]
    ) -> dict[str, EmbeddingRecord]:
        records: dict[str, EmbeddingRecord] = {}
        missing: list[str] = []
        for key in keys:
            cached = self._cache.get(key)
            expected_hash = source_hashes.get(key)
            if (
                cached is not None
                and cached.model_id == model_id
                and cached.source_hash == expected_hash
            ):
                self._cache.move_to_end(key)
                records[key] = cached
            else:
                missing.append(key)
        if not missing:
            return records
        placeholders = ", ".join("?" for _ in missing)
        with sqlite3.connect(self._path) as connection:
            rows = connection.execute(
                f"""
                SELECT key, vector, dimensions, model_id, source_hash, metadata
                FROM embeddings
                WHERE key IN ({placeholders}) AND model_id = ?
                """,
                [*missing, model_id],
            ).fetchall()
        for key, raw_vector, dimensions, stored_model_id, source_hash, raw_metadata in rows:
            if source_hashes.get(key) != source_hash:
                continue
            record = EmbeddingRecord(
                key=key,
                vector=tuple(float(value) for value in json.loads(raw_vector)),
                dimensions=int(dimensions),
                model_id=str(stored_model_id),
                source_hash=str(source_hash),
                metadata=dict(json.loads(raw_metadata)),
            )
            records[key] = record
            self._remember(record)
        return records

    def put_many(self, records: tuple[EmbeddingRecord, ...]) -> None:
        now = time.time()
        with sqlite3.connect(self._path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (key, vector, dimensions, model_id, source_hash, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.key,
                        json.dumps(list(record.vector)),
                        record.dimensions,
                        record.model_id,
                        record.source_hash,
                        now,
                        json.dumps(record.metadata, sort_keys=True),
                    )
                    for record in records
                ],
            )
            connection.commit()
        for record in records:
            self._remember(record)

    def get_or_embed(
        self,
        items: dict[str, str],
        *,
        backend: EmbeddingBackend,
        source_hashes: dict[str, str],
        metadata: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, EmbeddingRecord]:
        existing = self.get_many(tuple(items), model_id=backend.model_id, source_hashes=source_hashes)
        missing = [key for key in items if key not in existing]
        if not missing:
            return existing
        vectors = backend.embed(tuple(items[key] for key in missing))
        new_records = tuple(
            EmbeddingRecord(
                key=key,
                vector=vector,
                dimensions=len(vector),
                model_id=backend.model_id,
                source_hash=source_hashes[key],
                metadata=(metadata or {}).get(key, {}),
            )
            for key, vector in zip(missing, vectors, strict=True)
        )
        self.put_many(new_records)
        return {**existing, **{record.key: record for record in new_records}}

    def _remember(self, record: EmbeddingRecord) -> None:
        self._cache[record.key] = record
        self._cache.move_to_end(record.key)
        while len(self._cache) > self._max_cache_items:
            self._cache.popitem(last=False)

    def _initialize(self) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    PRIMARY KEY (key, model_id)
                )
                """
            )
            connection.commit()


def cosine_similarity(left: Vector, right: Vector) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
