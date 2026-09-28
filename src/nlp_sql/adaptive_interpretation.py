"""Interpretable adaptive query interpretation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp_sql.query_ast import QueryAST
from nlp_sql.result import PipelineResult


@dataclass(frozen=True)
class InterpretationCandidate:
    name: str
    ast: QueryAST | None
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ast": self.ast.to_dict() if self.ast is not None else None,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AdaptiveDecision:
    candidates: tuple[InterpretationCandidate, ...]
    selected_candidate: str | None
    confidence: float
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate": self.selected_candidate,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }


class AdaptiveInterpreter:
    """Rank parser outputs using explicit runtime and retrieval evidence."""

    def decide(self, result: PipelineResult) -> AdaptiveDecision:
        if result.ast is None:
            return AdaptiveDecision(
                candidates=(),
                selected_candidate=None,
                confidence=0.0,
                fallback_reason=result.error.code if result.error else "NO_AST",
            )
        retrieval_confidence = (
            float(result.schema_candidates[0]["score"]["confidence"])
            if result.schema_candidates
            else 0.0
        )
        parser_confidence = 0.90 if result.success else 0.0
        execution_signal = 0.05 if result.rows else 0.0
        confidence = min(1.0, parser_confidence * 0.7 + retrieval_confidence * 0.25 + execution_signal)
        candidate = InterpretationCandidate(
            name="deterministic_parser",
            ast=result.ast,
            confidence=confidence,
            evidence={
                "parser_success": result.success,
                "matched_rules": list(result.matched_rules),
                "schema_retrieval_confidence": retrieval_confidence,
                "execution_rows_observed": result.rows is not None,
            },
        )
        return AdaptiveDecision(
            candidates=(candidate,),
            selected_candidate=candidate.name,
            confidence=confidence,
            fallback_reason=None if result.success else "PARSER_REJECTED",
        )
