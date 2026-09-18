"""Typed Query AST used as the contract between parsing and SQL compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, cast


class QueryType(StrEnum):
    SELECT = "select"


class SortDirection(StrEnum):
    ASC = "ASC"
    DESC = "DESC"


@dataclass(frozen=True)
class ColumnExpression:
    type: str
    name: str


@dataclass(frozen=True)
class AggregationExpression:
    type: str
    function: str
    column: str
    alias: str | None = None


@dataclass(frozen=True)
class Predicate:
    column: str
    operator: str
    value: Any


@dataclass(frozen=True)
class DateRangePredicate:
    column: str
    start: date
    end: date


@dataclass(frozen=True)
class HavingPredicate:
    left: AggregationExpression
    operator: str
    value: Any


@dataclass(frozen=True)
class OrderBy:
    column: str | None = None
    direction: SortDirection = SortDirection.ASC
    aggregation: AggregationExpression | None = None


@dataclass(frozen=True)
class QueryAST:
    type: QueryType
    source: str
    select: tuple[ColumnExpression, ...] = ()
    aggregations: tuple[AggregationExpression, ...] = ()
    filters: tuple[Predicate | DateRangePredicate, ...] = ()
    having: tuple[HavingPredicate, ...] = ()
    group_by: tuple[str, ...] = ()
    order_by: tuple[OrderBy, ...] = ()
    limit: int | None = None
    status: str = "exact"
    matched_rules: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = cast(dict[str, Any], _json_safe(asdict(self)))
        data["type"] = self.type.value
        for order in data["order_by"]:
            order["direction"] = order["direction"].value
        return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
