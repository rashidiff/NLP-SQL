"""Benchmark data model for semantic Text-to-SQL evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nlp_sql.query_ast import QueryAST


class BenchmarkLabel(StrEnum):
    AMBIGUOUS = "ambiguous"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    PARAPHRASED = "paraphrased"
    SINGLE_TABLE = "single_table"
    MULTI_TABLE = "multi_table"


class FailureCategory(StrEnum):
    UNKNOWN_SCHEMA_CONCEPT = "unknown_schema_concept"
    AMBIGUOUS_NATURAL_LANGUAGE = "ambiguous_natural_language"
    WRONG_TABLE = "wrong_table"
    WRONG_COLUMN = "wrong_column"
    WRONG_OPERATOR = "wrong_operator"
    WRONG_AGGREGATION = "wrong_aggregation"
    WRONG_VALUE_INTERPRETATION = "wrong_value_interpretation"
    WRONG_ORDERING = "wrong_ordering"
    WRONG_LIMIT = "wrong_limit"
    UNSUPPORTED_CONSTRUCTION = "unsupported_construction"
    SEMANTICALLY_VALID_ALTERNATIVE = "semantically_valid_alternative"
    EXECUTION_FAILURE = "execution_failure"
    SAFETY_REJECTION = "safety_rejection"


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    dataset_id: str
    valid_asts: tuple[QueryAST, ...] = ()
    valid_sql: tuple[str, ...] = ()
    expected_result: tuple[dict[str, Any], ...] | None = None
    labels: tuple[BenchmarkLabel, ...] = ()
    failure_labels: tuple[FailureCategory, ...] = ()
    ordered: bool = False
    float_tolerance: float = 1e-6
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    success: bool
    matched_level: str | None
    failure_category: FailureCategory | None
    details: dict[str, Any]


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseEvaluation, ...]
    by_failure_category: dict[str, int]
    by_label: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": [
                {
                    "case_id": case.case_id,
                    "success": case.success,
                    "matched_level": case.matched_level,
                    "failure_category": case.failure_category.value
                    if case.failure_category is not None
                    else None,
                    "details": case.details,
                }
                for case in self.cases
            ],
            "by_failure_category": self.by_failure_category,
            "by_label": self.by_label,
        }
