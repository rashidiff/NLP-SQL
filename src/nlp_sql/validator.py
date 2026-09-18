"""AST validation before SQL compilation."""

from __future__ import annotations

from nlp_sql.query_ast import DateRangePredicate, Predicate, QueryAST
from nlp_sql.result import QueryError
from nlp_sql.schema import SchemaRegistry

SUPPORTED_AGGREGATIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
SUPPORTED_OPERATORS = {">", "<", ">=", "<=", "=", "!=", "BETWEEN"}
MAX_LIMIT = 1000


class ASTValidator:
    def __init__(self, schema: SchemaRegistry, max_limit: int = MAX_LIMIT) -> None:
        self._schema = schema
        self._max_limit = max_limit

    def validate(self, ast: QueryAST) -> QueryError | None:
        if not self._schema.has_table(ast.source):
            return QueryError("UNKNOWN_TABLE", f"Table '{ast.source}' is not configured.")
        if ast.limit is not None and not (1 <= ast.limit <= self._max_limit):
            return QueryError("INVALID_LIMIT", f"Limit must be between 1 and {self._max_limit}.")
        for column in ast.select:
            if column.name != "*" and not self._schema.has_column(ast.source, column.name):
                return QueryError(
                    "UNKNOWN_COLUMN", f"Column '{column.name}' is not valid for {ast.source}."
                )
        for aggregation in ast.aggregations:
            if aggregation.function not in SUPPORTED_AGGREGATIONS:
                return QueryError(
                    "UNSUPPORTED_AGGREGATION", f"Unsupported aggregation {aggregation.function}."
                )
            if aggregation.column != "*" and not self._schema.has_column(
                ast.source, aggregation.column
            ):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Column '{aggregation.column}' is not valid for {ast.source}.",
                )
        for predicate in ast.filters:
            error = self._validate_predicate(ast.source, predicate)
            if error is not None:
                return error
        for having_predicate in ast.having:
            if having_predicate.operator not in SUPPORTED_OPERATORS:
                return QueryError(
                    "UNSUPPORTED_OPERATOR",
                    f"Unsupported operator {having_predicate.operator}.",
                )
            if having_predicate.left.function not in SUPPORTED_AGGREGATIONS:
                return QueryError(
                    "UNSUPPORTED_AGGREGATION",
                    f"Unsupported aggregation {having_predicate.left.function}.",
                )
            if not self._schema.has_column(ast.source, having_predicate.left.column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Column '{having_predicate.left.column}' is not valid for {ast.source}.",
                )
        for group_column in ast.group_by:
            if not self._schema.has_column(ast.source, group_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Group by column '{group_column}' is not valid for {ast.source}.",
                )
        for order in ast.order_by:
            if order.column is not None and not self._schema.has_column(ast.source, order.column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Order by column '{order.column}' is not valid for {ast.source}.",
                )
        if ast.having and not ast.group_by:
            return QueryError("INVALID_HAVING", "HAVING requires GROUP BY in this engine.")
        return None

    def _validate_predicate(
        self, source: str, predicate: Predicate | DateRangePredicate
    ) -> QueryError | None:
        if isinstance(predicate, DateRangePredicate):
            if not self._schema.has_column(source, predicate.column):
                return QueryError(
                    "UNKNOWN_COLUMN", f"Date column '{predicate.column}' is not valid for {source}."
                )
            return None
        if not self._schema.has_column(source, predicate.column):
            return QueryError(
                "UNKNOWN_COLUMN", f"Column '{predicate.column}' is not valid for {source}."
            )
        if predicate.operator not in SUPPORTED_OPERATORS:
            return QueryError("UNSUPPORTED_OPERATOR", f"Unsupported operator {predicate.operator}.")
        return None
