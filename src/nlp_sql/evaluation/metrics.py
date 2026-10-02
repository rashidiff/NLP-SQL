"""Semantic, SQL, and execution-equivalence metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nlp_sql.query_ast import DateRangePredicate, Predicate, QueryAST
from nlp_sql.result import SqlQuery


@dataclass(frozen=True)
class PrecisionRecallF1:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}


def schema_linking_metrics(candidate: QueryAST, reference: QueryAST) -> dict[str, object]:
    candidate_items = _schema_items(candidate)
    reference_items = _schema_items(reference)
    return {
        "tables": _score_sets({candidate.source}, {reference.source}).to_dict(),
        "columns": _score_sets(candidate_items["columns"], reference_items["columns"]).to_dict(),
        "aggregations": _score_sets(
            candidate_items["aggregations"], reference_items["aggregations"]
        ).to_dict(),
        "filters": _score_sets(candidate_items["filters"], reference_items["filters"]).to_dict(),
    }


def ast_structurally_equivalent(candidate: QueryAST, reference: QueryAST) -> bool:
    candidate_data = candidate.to_dict()
    reference_data = reference.to_dict()
    candidate_data.pop("matched_rules", None)
    reference_data.pop("matched_rules", None)
    return candidate_data == reference_data


def normalize_sql(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    normalized = normalized.replace(" ,", ",")
    return normalized


def sql_equivalent(candidate: SqlQuery | str, reference: SqlQuery | str) -> bool:
    candidate_sql = candidate.sql if isinstance(candidate, SqlQuery) else candidate
    reference_sql = reference.sql if isinstance(reference, SqlQuery) else reference
    return normalize_sql(candidate_sql) == normalize_sql(reference_sql)


def result_sets_equivalent(
    candidate_rows: tuple[dict[str, Any], ...],
    reference_rows: tuple[dict[str, Any], ...],
    *,
    ordered: bool,
    float_tolerance: float,
) -> bool:
    if ordered:
        return _rows_equal(candidate_rows, reference_rows, float_tolerance=float_tolerance)
    return sorted(_row_key(row) for row in candidate_rows) == sorted(
        _row_key(row) for row in reference_rows
    )


def _schema_items(ast: QueryAST) -> dict[str, set[str]]:
    columns = {column.name for column in ast.select if column.name != "*"}
    columns.update(aggregation.column for aggregation in ast.aggregations)
    columns.update(order.column for order in ast.order_by if order.column)
    filters: set[str] = set()
    for predicate in ast.filters:
        columns.add(predicate.column)
        if isinstance(predicate, DateRangePredicate):
            filters.add(f"{predicate.column}:range")
        elif isinstance(predicate, Predicate):
            filters.add(f"{predicate.column}:{predicate.operator}:{predicate.value}")
    return {
        "columns": columns,
        "aggregations": {
            f"{aggregation.function}:{aggregation.column}" for aggregation in ast.aggregations
        },
        "filters": filters,
    }


def _score_sets(candidate: set[str], reference: set[str]) -> PrecisionRecallF1:
    if not candidate and not reference:
        return PrecisionRecallF1(1.0, 1.0, 1.0)
    overlap = candidate & reference
    precision = len(overlap) / len(candidate) if candidate else 0.0
    recall = len(overlap) / len(reference) if reference else 0.0
    if precision + recall == 0:
        return PrecisionRecallF1(precision, recall, 0.0)
    return PrecisionRecallF1(precision, recall, 2 * precision * recall / (precision + recall))


def _rows_equal(
    left: tuple[dict[str, Any], ...],
    right: tuple[dict[str, Any], ...],
    *,
    float_tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if set(left_row) != set(right_row):
            return False
        for key, left_value in left_row.items():
            right_value = right_row[key]
            if isinstance(left_value, float) or isinstance(right_value, float):
                if abs(float(left_value) - float(right_value)) > float_tolerance:
                    return False
            elif left_value != right_value:
                return False
    return True


def _row_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in row.items()))
