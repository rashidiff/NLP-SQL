"""Non-LLM semantic planning with local candidate search and exploration.

The planner deliberately produces QueryAST candidates through the existing
safe parser contract. It adds semantic paraphrase expansion, schema-aware
ranking, and stochastic exploration without generating SQL or calling an LLM.
"""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass

from nlp_sql.embedding_parser import EmbeddingParser
from nlp_sql.feedback import QueryMemory, plan_signature
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.retrieval.base import SemanticSchemaRetriever
from nlp_sql.schema import SchemaRegistry


@dataclass(frozen=True)
class SemanticPlannerConfig:
    beam_width: int = 6
    temperature: float = 0.18
    clarification_margin: float = 0.035

    @classmethod
    def from_environment(cls) -> SemanticPlannerConfig:
        return cls(
            beam_width=max(2, int(os.getenv("NLP_SQL_SEMANTIC_BEAM_WIDTH", "6"))),
            temperature=max(0.0, float(os.getenv("NLP_SQL_SEMANTIC_TEMPERATURE", "0.18"))),
            clarification_margin=max(
                0.0, float(os.getenv("NLP_SQL_SEMANTIC_CLARIFICATION_MARGIN", "0.035"))
            ),
        )


class SemanticPlanner:
    """Select among locally generated AST candidates without an LLM."""

    _SYNONYMS = {
        "costly": "expensive",
        "pricey": "expensive",
        "affordable": "cheap",
        "properties": "homes",
        "houses": "homes",
        "lots": "land",
        "acreage": "land",
        "bed": "bedroom",
        "beds": "bedrooms",
        "latest": "recent",
        "newest": "recent",
        "largest": "large",
        "biggest": "large",
    }
    _STRONG_CUES = {
        "cheap",
        "expensive",
        "large",
        "many",
        "recent",
        "latest",
        "newest",
        "oldest",
        "top",
        "bottom",
        "average",
        "avg",
        "sum",
        "total",
        "count",
        "maximum",
        "minimum",
    }

    def __init__(
        self,
        schema: SchemaRegistry,
        parser: EmbeddingParser,
        retriever: SemanticSchemaRetriever,
        memory: QueryMemory | None = None,
        config: SemanticPlannerConfig | None = None,
    ) -> None:
        self._schema = schema
        self._parser = parser
        self._retriever = retriever
        self._memory = memory
        self._config = config or SemanticPlannerConfig.from_environment()
        self._random = random.SystemRandom()

    def plan(self, query: str) -> PipelineResult:
        candidates: list[tuple[PipelineResult, float]] = []
        seen: set[str] = set()
        for variant in self._variants(query):
            result = self._parser.parse(variant)
            if not result.success or result.ast is None:
                continue
            signature = repr(result.ast.to_dict())
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((result, self._score(query, result)))
        if not candidates:
            return PipelineResult(
                success=False,
                error=QueryError("SEMANTIC_NO_PLAN", "No local semantic plan was found."),
            )

        candidates.sort(key=lambda item: item[1], reverse=True)
        beam = candidates[: self._config.beam_width]
        if len(beam) > 1 and self._is_ambiguous(query, beam):
            return PipelineResult(
                success=False,
                requires_clarification=True,
                error=QueryError(
                    "AMBIGUOUS_QUERY",
                    "Multiple local semantic interpretations have similar confidence.",
                ),
                confidence=self._probability(beam[0][1], [score for _, score in beam]),
                candidate_count=len(beam),
                trace={"planner": "local-semantic", "beam_width": len(beam)},
            )

        scores = [score for _, score in beam]
        selected_index = self._sample_index(scores)
        selected, selected_score = beam[selected_index]
        probability = self._probability(selected_score, scores)
        interpretation = dict(selected.interpretation or {})
        interpretation.update(
            {
                "parser": "local-semantic",
                "semantic_probability": probability,
                "semantic_variant": selected.interpretation.get("variant")
                if selected.interpretation
                else None,
            }
        )
        return PipelineResult(
            success=True,
            ast=selected.ast,
            interpretation=interpretation,
            matched_rules=selected.matched_rules + ("planner:local-semantic",),
            confidence=probability,
            candidate_count=len(beam),
            trace={
                "planner": "local-semantic",
                "beam_width": len(beam),
                "temperature": self._config.temperature,
                "candidate_scores": [round(score, 6) for score in scores],
            },
        )

    def _variants(self, query: str) -> tuple[str, ...]:
        normalized = query.lower().strip()
        variants = [normalized]
        replaced = normalized
        for source, target in self._SYNONYMS.items():
            replaced = re.sub(rf"\b{re.escape(source)}\b", target, replaced)
        if replaced != normalized:
            variants.append(replaced)
        tokens = normalized.split()
        if len(tokens) > 2:
            variants.append(" ".join(tokens[::-1]))
        return tuple(dict.fromkeys(variants))

    def _score(self, query: str, result: PipelineResult) -> float:
        assert result.ast is not None
        retrieved = self._retriever.search(query, top_k=8)
        source_score = max(
            (
                candidate.score.final
                for candidate in retrieved
                if candidate.document.table == result.ast.source
            ),
            default=0.0,
        )
        column_names = {
            str(getattr(column, "name", getattr(column, "column", "")))
            for column in result.ast.select
        }
        column_names.update(aggregation.column for aggregation in result.ast.aggregations)
        column_score = max(
            (
                candidate.score.final
                for candidate in retrieved
                if candidate.document.column in column_names
            ),
            default=0.0,
        )
        complexity_penalty = 0.02 * max(0, len(result.ast.joins) - 2)
        memory_prior = (
            self._memory.success_prior(query, plan_signature(result.ast)) if self._memory else 0.0
        )
        exploration = self._random.random() * self._config.temperature * 0.04
        return max(
            0.0,
            0.50 * source_score
            + 0.32 * column_score
            + 0.10
            + 0.08 * memory_prior
            - complexity_penalty
            + exploration,
        )

    def _is_ambiguous(self, query: str, beam: list[tuple[PipelineResult, float]]) -> bool:
        if set(re.findall(r"[a-z0-9_]+", query.lower())) & self._STRONG_CUES:
            return False
        first, second = beam[0], beam[1]
        if first[0].ast is None or second[0].ast is None:
            return False
        return (
            abs(first[1] - second[1]) <= self._config.clarification_margin
            and first[0].ast.to_dict() != second[0].ast.to_dict()
        )

    def _sample_index(self, scores: list[float]) -> int:
        if self._config.temperature <= 0 or len(scores) == 1:
            return 0
        peak = max(scores)
        weights = [math.exp((score - peak) / self._config.temperature) for score in scores]
        return self._random.choices(range(len(scores)), weights=weights, k=1)[0]

    @staticmethod
    def _probability(score: float, scores: list[float]) -> float:
        peak = max(scores)
        weights = [math.exp(value - peak) for value in scores]
        return math.exp(score - peak) / sum(weights)
