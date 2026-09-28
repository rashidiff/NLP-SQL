"""Hybrid lexical and local-embedding schema retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nlp_sql.retrieval.adaptive import AdaptiveVectorIndex
from nlp_sql.retrieval.base import (
    CandidateKind,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalScore,
    SchemaDocument,
)
from nlp_sql.retrieval.embeddings import (
    EmbeddingBackend,
    EmbeddingStore,
    LocalHashEmbeddingBackend,
)
from nlp_sql.retrieval.lexical import LexicalSchemaRetriever
from nlp_sql.retrieval.vector_index import InMemoryVectorIndex


@dataclass
class HybridSchemaRetriever:
    documents: tuple[SchemaDocument, ...]
    config: RetrievalConfig = RetrievalConfig()
    embedding_backend: EmbeddingBackend = field(default_factory=LocalHashEmbeddingBackend)
    embedding_store: EmbeddingStore | None = None
    vector_index: AdaptiveVectorIndex = field(
        default_factory=lambda: AdaptiveVectorIndex(InMemoryVectorIndex())
    )

    @classmethod
    def with_sqlite_store(
        cls, documents: tuple[SchemaDocument, ...], path: Path, config: RetrievalConfig | None = None
    ) -> HybridSchemaRetriever:
        return cls(
            documents=documents,
            config=config or RetrievalConfig(),
            embedding_store=EmbeddingStore(path),
        )

    def __post_init__(self) -> None:
        self._lexical = LexicalSchemaRetriever(self.documents, self.config)
        self._by_key = {document.key: document for document in self.documents}
        self._ensure_indexed()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        parser_evidence: dict[str, float] | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        limit = top_k or self.config.top_k
        lexical = {
            candidate.document.key: candidate
            for candidate in self._lexical.search(
                query, top_k=max(limit * 2, limit), parser_evidence=parser_evidence
            )
        }
        query_vector = self.embedding_backend.embed((query,))[0]
        vector_results = self.vector_index.search(query, query_vector, top_k=max(limit * 2, limit))
        keys = set(lexical) | {result.key for result in vector_results}
        embedding_scores = {result.key: max(result.score, 0.0) for result in vector_results}
        ranked: list[RetrievalCandidate] = []
        for key in keys:
            document = self._by_key.get(key)
            if document is None:
                continue
            lexical_score = lexical.get(key).score.lexical if key in lexical else 0.0
            embedding_score = embedding_scores.get(key, 0.0)
            parser_score = (parser_evidence or {}).get(key, 0.0)
            type_score = self._type_compatibility(query, document)
            final = min(
                1.0,
                lexical_score * self.config.lexical_weight
                + embedding_score * self.config.embedding_weight
                + type_score * self.config.type_weight
                + parser_score * self.config.parser_weight,
            )
            if final < self.config.confidence_threshold:
                continue
            ranked.append(
                RetrievalCandidate(
                    document=document,
                    score=RetrievalScore(
                        lexical=lexical_score,
                        embedding=embedding_score,
                        type_compatibility=type_score,
                        parser_evidence=parser_score,
                        final=final,
                        confidence=final,
                    ),
                    evidence=self._evidence(key, lexical, embedding_score, type_score),
                )
            )
        ranked.sort(key=lambda candidate: candidate.score.final, reverse=True)
        return tuple(ranked[:limit])

    def stats(self) -> dict[str, object]:
        return self.vector_index.stats()

    def _ensure_indexed(self) -> None:
        items = {document.key: document.text for document in self.documents}
        source_hashes = {document.key: document.source_hash for document in self.documents}
        metadata = {
            document.key: {
                "kind": document.kind.value,
                "table": document.table or "",
                "column": document.column or "",
            }
            for document in self.documents
        }
        if self.embedding_store is None:
            vectors = self.embedding_backend.embed(tuple(items.values()))
            for key, vector in zip(items, vectors, strict=True):
                self.vector_index.insert(key, vector)
            return
        records = self.embedding_store.get_or_embed(
            items,
            backend=self.embedding_backend,
            source_hashes=source_hashes,
            metadata=metadata,
        )
        for key, record in records.items():
            self.vector_index.insert(key, record.vector)

    def _type_compatibility(self, query: str, document: SchemaDocument) -> float:
        if document.kind != CandidateKind.COLUMN:
            return 0.25
        text = query.lower()
        if document.data_type in {"integer", "decimal"} and any(
            cue in text for cue in ("top", "bottom", "highest", "lowest", "average", "sum", "total")
        ):
            return 1.0
        if document.data_type == "datetime" and any(
            cue in text for cue in ("recent", "latest", "oldest", "before", "after", "date")
        ):
            return 1.0
        if document.data_type == "string" and any(cue in text for cue in ("where", "type", "use")):
            return 0.75
        return 0.35

    def _evidence(
        self,
        key: str,
        lexical: dict[str, RetrievalCandidate],
        embedding_score: float,
        type_score: float,
    ) -> tuple[str, ...]:
        evidence: list[str] = []
        if key in lexical:
            evidence.extend(lexical[key].evidence)
        if embedding_score > 0:
            evidence.append(f"embedding:{embedding_score:.3f}")
        if type_score > 0:
            evidence.append(f"type:{type_score:.2f}")
        return tuple(evidence)
