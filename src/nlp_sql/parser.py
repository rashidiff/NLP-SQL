"""Deterministic rule-based semantic parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from nlp_sql.date_parser import DateExpressionParser
from nlp_sql.normalizer import TextNormalizer
from nlp_sql.query_ast import (
    AggregationExpression,
    ColumnExpression,
    DateRangePredicate,
    HavingPredicate,
    OrderBy,
    Predicate,
    QueryAST,
    QueryType,
    SortDirection,
)
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry
from nlp_sql.tokenizer import PhraseKind, PhraseMatch, PhraseMatcher, Tokenizer


@dataclass(frozen=True)
class SemanticParse:
    ast: QueryAST
    interpretation: dict[str, object]


class RuleBasedParser:
    """Small, deterministic grammar for supported business queries."""

    _AGGREGATION_RULES: tuple[tuple[str, str, str], ...] = (
        ("how many", "COUNT", "id"),
        ("number of", "COUNT", "id"),
        ("count", "COUNT", "id"),
        ("total revenue", "SUM", "amount"),
        ("total sales", "SUM", "amount"),
        ("total spending", "SUM", "amount"),
        ("sum", "SUM", "amount"),
        ("average order amount", "AVG", "amount"),
        ("average spending", "AVG", "amount"),
        ("average", "AVG", "amount"),
        ("maximum order value", "MAX", "amount"),
        ("maximum", "MAX", "amount"),
        ("minimum order value", "MIN", "amount"),
        ("minimum", "MIN", "amount"),
    )
    _OPERATOR_WORDS = {
        "above": ">",
        "below": "<",
        "minimum": ">=",
        "maximum": "<=",
        "equals": "=",
        "is": "=",
        "after": ">",
        "before": "<",
    }
    _DATE_BOUND_RE = re.compile(
        r"\b(?:after|before)\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\b"
    )

    def __init__(
        self,
        schema: SchemaRegistry,
        today: date | None = None,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self._schema = schema
        self._normalizer = normalizer or TextNormalizer()
        self._tokenizer = Tokenizer()
        self._matcher = PhraseMatcher()
        self._date_parser = DateExpressionParser(today=today)
        self._safety = SafetyValidator()

    def parse(self, query: str) -> PipelineResult:
        normalized = self._normalizer.normalize(query)
        safety_error = self._safety.validate_input(normalized.text)
        if safety_error is not None:
            return PipelineResult(success=False, error=safety_error)

        if self._requires_threshold(normalized.text):
            return PipelineResult(
                success=False,
                requires_clarification=True,
                error=QueryError(
                    "MISSING_THRESHOLD",
                    "What amount should be considered high spending?",
                ),
            )

        tokens = self._tokenizer.tokenize(normalized.text)
        phrase_matches = self._matcher.find_matches(tokens)
        matched_rules = [f"{match.kind.value}:{match.phrase}" for match in phrase_matches]

        source = self._resolve_source(normalized.text)
        if source is None:
            return PipelineResult(
                success=False,
                error=QueryError(
                    "UNSUPPORTED_QUERY",
                    "Could not resolve a supported source table.",
                ),
            )
        matched_rules.append(f"source:{source}")

        aggregation = self._detect_aggregation(normalized.text, source)
        if aggregation is not None:
            matched_rules.append(f"aggregation:{aggregation.function}")

        select = self._build_select(normalized.text, source, aggregation)
        group_by = self._detect_group_by(normalized.text, source, aggregation)
        amount_predicate = self._extract_amount_predicate(normalized.text, phrase_matches)
        filters: list[Predicate | DateRangePredicate] = []
        having: list[HavingPredicate] = []
        if amount_predicate is not None:
            if (
                group_by
                and aggregation is not None
                and aggregation.function in {"SUM", "AVG", "MIN", "MAX"}
            ):
                having.append(
                    HavingPredicate(
                        left=AggregationExpression("aggregation", aggregation.function, "amount"),
                        operator=amount_predicate.operator,
                        value=amount_predicate.value,
                    )
                )
            else:
                filters.append(amount_predicate)
            matched_rules.append("amount_filter")

        for date_range in self._date_parser.extract_ranges(normalized.text):
            column = self._date_column(source)
            if self._schema.has_column(source, column):
                filters.append(DateRangePredicate(column, date_range.start, date_range.end))
                matched_rules.append(f"date:{date_range.rule}")

        limit = self._detect_limit(normalized.text)
        order_by = self._detect_order_by(normalized.text, aggregation, limit)
        if limit is not None:
            matched_rules.append("limit")

        ast = QueryAST(
            type=QueryType.SELECT,
            source=source,
            select=tuple(select),
            aggregations=(aggregation,) if aggregation is not None else (),
            filters=tuple(filters),
            having=tuple(having),
            group_by=tuple(group_by),
            order_by=tuple(order_by),
            limit=limit,
            status="inferred"
            if group_by or aggregation or filters or order_by or limit
            else "exact",
            matched_rules=tuple(matched_rules),
        )
        return PipelineResult(
            success=True,
            ast=ast,
            interpretation=self._interpret(ast, normalized.text),
            matched_rules=ast.matched_rules,
        )

    def _resolve_source(self, text: str) -> str | None:
        if (
            "spent" in text or "spending" in text or "revenue" in text or "sales" in text
        ) and self._schema.has_table("orders"):
            return "orders"
        for phrase in _phrases(text, max_words=3):
            resolved = self._schema.resolve_table(phrase)
            if resolved is not None:
                return resolved.canonical
        column_tables = {
            candidate.table
            for phrase in _phrases(text, max_words=3)
            for candidate in self._schema.resolve_column(phrase)
        }
        if len(column_tables) == 1:
            return next(iter(column_tables))
        return None

    def _detect_aggregation(self, text: str, source: str) -> AggregationExpression | None:
        for phrase, function, column in self._AGGREGATION_RULES:
            if phrase in text:
                if function == "COUNT":
                    column = self._count_column(source)
                elif resolved_column := self._resolve_column_in_text(text, source):
                    column = resolved_column
                elif not self._schema.has_column(source, column):
                    column = self._default_metric_column(source) or column
                alias = _aggregation_alias(function, column)
                return AggregationExpression("aggregation", function, column, alias)
        if "spent" in text or "spending" in text:
            return AggregationExpression("aggregation", "SUM", "amount", "total_purchase")
        return None

    def _build_select(
        self, text: str, source: str, aggregation: AggregationExpression | None
    ) -> list[ColumnExpression]:
        if aggregation is not None and "customer" in text and source == "orders":
            return [ColumnExpression("column", "customer_id")]
        if aggregation is not None:
            return []
        return [ColumnExpression("column", "*")]

    def _detect_group_by(
        self, text: str, source: str, aggregation: AggregationExpression | None
    ) -> list[str]:
        if any(keyword in text for keyword in ("top", "bottom", "order by", "sort by")):
            return []
        generic_group = re.search(r"\b(?:by|per)\s+([a-z_ ]+)$", text)
        if generic_group:
            column = self._resolve_column_phrase(generic_group.group(1), source)
            if column is not None:
                return [column]
        mentions_customer_group = (
            " per customer" in text
            or " by customer" in text
            or ("customers" in text and "spent" in text)
            or ("customers" in text and aggregation is not None and source == "orders")
        )
        if mentions_customer_group and source == "orders":
            return ["customer_id"]
        if " by month" in text:
            return ["created_at"]
        return []

    def _extract_amount_predicate(
        self, text: str, phrase_matches: tuple[PhraseMatch, ...]
    ) -> Predicate | None:
        if self._DATE_BOUND_RE.search(text):
            return None
        between = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\b", text)
        if between:
            return Predicate(
                "amount",
                "BETWEEN",
                (float(between.group(1)), float(between.group(2))),
            )

        operator = None
        for match in phrase_matches:
            if match.kind == PhraseKind.OPERATOR:
                operator = match.value
                break
        if operator is None:
            for word, mapped in self._OPERATOR_WORDS.items():
                if f" {word} " in f" {text} ":
                    operator = mapped
                    break
        if operator is None:
            return None
        operator_pattern = "|".join(map(re.escape, self._OPERATOR_WORDS))
        value_match = re.search(
            rf"(?:{operator_pattern}|than|to)\s+(\d+(?:\.\d+)?)",
            text,
        )
        if value_match is None:
            value_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if value_match is None:
            return None
        value_text = value_match.group(1)
        value: int | float = int(value_text) if value_text.isdigit() else float(value_text)
        column = self._predicate_column(text)
        if column is None:
            return None
        return Predicate(column, operator, value)

    def _detect_order_by(
        self, text: str, aggregation: AggregationExpression | None, limit: int | None
    ) -> list[OrderBy]:
        direction = (
            SortDirection.ASC
            if "ascending" in text or "oldest" in text or "lowest" in text or "bottom" in text
            else SortDirection.DESC
        )
        if "newest" in text or "latest" in text or "oldest" in text:
            return [OrderBy(column="created_at", direction=direction)]
        if (
            "highest" in text
            or "lowest" in text
            or "descending" in text
            or "ascending" in text
            or ("top" in text and limit is not None)
            or ("bottom" in text and limit is not None)
        ):
            source = self._resolve_source(text)
            resolved_column = self._resolve_column_in_text(text, source) if source else None
            if aggregation is not None and (
                "spending" in text or "revenue" in text or "sales" in text
            ):
                return [OrderBy(direction=direction, aggregation=aggregation)]
            if resolved_column is not None:
                return [OrderBy(column=resolved_column, direction=direction)]
            if "name" in text:
                return [OrderBy(column="name", direction=direction)]
            return [
                OrderBy(column=self._default_metric_column_from_text(text), direction=direction)
            ]
        if "order by customer name" in text or "sort customers by name" in text:
            return [OrderBy(column="name", direction=direction)]
        return []

    def _detect_limit(self, text: str) -> int | None:
        match = re.search(r"\b(?:top|bottom|first|latest|oldest)\s+(\d+)\b", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d+)\s+highest", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d+)\s+\w+\s+with\s+the\s+(?:highest|lowest)\b", text)
        if match:
            return int(match.group(1))
        return None

    def _requires_threshold(self, text: str) -> bool:
        if "recent customers" in text:
            return True
        high_without_measure = ("high value" in text or "high spending" in text) and not re.search(
            r"\b\d+\b|\bhighest\b|\btop\b",
            text,
        )
        return high_without_measure

    def _interpret(self, ast: QueryAST, text: str) -> dict[str, object]:
        interpretation: dict[str, object] = {"source": ast.source, "status": ast.status}
        if "customer" in text:
            interpretation["entity"] = "customers"
        if ast.aggregations:
            interpretation["aggregation"] = ast.aggregations[0].function
            interpretation["metric"] = ast.aggregations[0].column
        if ast.group_by:
            interpretation["group_by"] = ast.group_by[0]
        if ast.having:
            interpretation["operator"] = ast.having[0].operator
            interpretation["value"] = ast.having[0].value
        elif ast.filters:
            for predicate in ast.filters:
                if isinstance(predicate, Predicate):
                    interpretation["operator"] = predicate.operator
                    interpretation["value"] = predicate.value
                    interpretation["metric"] = predicate.column
                    break
        return interpretation

    def _count_column(self, source: str) -> str:
        for candidate in ("id", "unique_id"):
            if self._schema.has_column(source, candidate):
                return candidate
        table = self._schema.tables[source]
        return next(iter(table.columns))

    def _default_metric_column(self, source: str) -> str | None:
        for candidate in ("amount", "sale_price", "total_value", "value", "price"):
            if self._schema.has_column(source, candidate):
                return candidate
        table = self._schema.tables.get(source)
        if table is None:
            return None
        for column in table.columns.values():
            if column.type in {"integer", "decimal"}:
                return column.name
        return None

    def _default_metric_column_from_text(self, text: str) -> str:
        source = self._resolve_source(text)
        if source is None:
            return "amount"
        return self._default_metric_column(source) or "amount"

    def _predicate_column(self, text: str) -> str | None:
        source = self._resolve_source(text)
        if source is None:
            return None
        if resolved := self._resolve_column_in_text(text, source):
            return resolved
        return self._default_metric_column(source)

    def _resolve_column_in_text(self, text: str, source: str) -> str | None:
        for phrase in _phrases(text, max_words=3):
            column = self._resolve_column_phrase(phrase, source)
            if column is not None:
                return column
        return None

    def _resolve_column_phrase(self, phrase: str, source: str) -> str | None:
        for candidate in self._schema.resolve_column(phrase, table=source):
            return candidate.column
        return None

    def _date_column(self, source: str) -> str:
        for candidate in ("created_at", "sale_date", "date"):
            if self._schema.has_column(source, candidate):
                return candidate
        return "created_at"


def _phrases(text: str, max_words: int) -> tuple[str, ...]:
    tokens = text.split()
    phrases: list[str] = []
    for size in range(max_words, 0, -1):
        for index in range(0, len(tokens) - size + 1):
            phrases.append(" ".join(tokens[index : index + size]))
    return tuple(phrases)


def _aggregation_alias(function: str, column: str) -> str:
    if function == "COUNT":
        return "count"
    if function == "SUM" and column == "amount":
        return "total_amount"
    return f"{function.lower()}_{column}"
