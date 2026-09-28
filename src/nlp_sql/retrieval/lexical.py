"""Lexical schema retrieval using aliases and token overlap."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_sql.retrieval.base import (
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalScore,
    SchemaDocument,
    tokenize,
)


@dataclass(frozen=True)
class LexicalSchemaRetriever:
    documents: tuple[SchemaDocument, ...]
    config: RetrievalConfig = RetrievalConfig()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        parser_evidence: dict[str, float] | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        query_tokens = set(tokenize(query))
        limit = top_k or self.config.top_k
        candidates: list[RetrievalCandidate] = []
        for document in self.documents:
            score, evidence = self._score_document(query, query_tokens, document)
            if parser_evidence:
                parser_score = parser_evidence.get(document.key, 0.0)
            else:
                parser_score = 0.0
            final = min(1.0, score + parser_score * self.config.parser_weight)
            if final <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    document=document,
                    score=RetrievalScore(
                        lexical=score,
                        parser_evidence=parser_score,
                        final=final,
                        confidence=final,
                    ),
                    evidence=tuple(evidence),
                )
            )
        return tuple(
            sorted(candidates, key=lambda candidate: candidate.score.final, reverse=True)[:limit]
        )

    def _score_document(
        self, query: str, query_tokens: set[str], document: SchemaDocument
    ) -> tuple[float, list[str]]:
        doc_tokens = set(tokenize(document.text))
        if not query_tokens or not doc_tokens:
            return 0.0, []
        overlap = query_tokens & doc_tokens
        score = len(overlap) / max(len(query_tokens), 1)
        evidence = [f"token:{token}" for token in sorted(overlap)]
        for alias in (document.physical_name, document.human_name, *document.aliases):
            normalized = alias.lower().replace("_", " ")
            if normalized and normalized in query.lower():
                score = max(score, 0.85)
                evidence.append(f"alias:{normalized}")
        return min(score, 1.0), evidence
