"""Evaluation runner for ambiguity-aware Text-to-SQL benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from nlp_sql.engine import NlpSqlEngine
from nlp_sql.evaluation.benchmark import (
    BenchmarkCase,
    CaseEvaluation,
    EvaluationReport,
    FailureCategory,
)
from nlp_sql.evaluation.metrics import (
    ast_structurally_equivalent,
    schema_linking_metrics,
    sql_equivalent,
)


@dataclass(frozen=True)
class EvaluationRunner:
    engine: NlpSqlEngine

    def evaluate(self, cases: tuple[BenchmarkCase, ...]) -> EvaluationReport:
        results: list[CaseEvaluation] = []
        failures: Counter[str] = Counter()
        by_label: dict[str, Counter[str]] = defaultdict(Counter)
        for case in cases:
            result = self.engine.parse(case.question)
            evaluated = self._evaluate_case(case, result)
            results.append(evaluated)
            if evaluated.failure_category is not None:
                failures[evaluated.failure_category.value] += 1
            for label in case.labels:
                by_label[label.value]["total"] += 1
                by_label[label.value]["success" if evaluated.success else "failure"] += 1
        return EvaluationReport(
            cases=tuple(results),
            by_failure_category=dict(failures),
            by_label={label: dict(counter) for label, counter in by_label.items()},
        )

    def _evaluate_case(self, case: BenchmarkCase, result: object) -> CaseEvaluation:
        if not getattr(result, "success", False):
            return CaseEvaluation(
                case_id=case.id,
                success=False,
                matched_level=None,
                failure_category=FailureCategory.SAFETY_REJECTION
                if getattr(getattr(result, "error", None), "code", "") in {"FORBIDDEN_OPERATION", "UNSAFE_SQL"}
                else FailureCategory.UNSUPPORTED_CONSTRUCTION,
                details={"error": getattr(getattr(result, "error", None), "code", None)},
            )
        candidate_ast = getattr(result, "ast", None)
        candidate_sql = getattr(result, "sql", None)
        for reference_ast in case.valid_asts:
            if candidate_ast is not None and ast_structurally_equivalent(candidate_ast, reference_ast):
                return CaseEvaluation(case.id, True, "ast", None, {"matched_reference": "ast"})
        for reference_sql in case.valid_sql:
            if candidate_sql is not None and sql_equivalent(candidate_sql, reference_sql):
                return CaseEvaluation(case.id, True, "sql_normalized", None, {})
        if case.valid_asts and candidate_ast is not None:
            best_metrics = schema_linking_metrics(candidate_ast, case.valid_asts[0])
            return CaseEvaluation(
                case_id=case.id,
                success=False,
                matched_level="schema_linking",
                failure_category=self._failure_from_schema_metrics(best_metrics),
                details={"schema_linking": best_metrics},
            )
        return CaseEvaluation(
            case_id=case.id,
            success=False,
            matched_level=None,
            failure_category=FailureCategory.UNSUPPORTED_CONSTRUCTION,
            details={},
        )

    def _failure_from_schema_metrics(self, metrics: dict[str, object]) -> FailureCategory:
        table_f1 = metrics["tables"]["f1"]  # type: ignore[index]
        column_f1 = metrics["columns"]["f1"]  # type: ignore[index]
        aggregation_f1 = metrics["aggregations"]["f1"]  # type: ignore[index]
        filter_f1 = metrics["filters"]["f1"]  # type: ignore[index]
        if table_f1 < 1:
            return FailureCategory.WRONG_TABLE
        if aggregation_f1 < 1:
            return FailureCategory.WRONG_AGGREGATION
        if filter_f1 < 1:
            return FailureCategory.WRONG_OPERATOR
        if column_f1 < 1:
            return FailureCategory.WRONG_COLUMN
        return FailureCategory.SEMANTICALLY_VALID_ALTERNATIVE
