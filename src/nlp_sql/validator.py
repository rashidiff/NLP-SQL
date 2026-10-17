"""AST validation before SQL compilation."""

from __future__ import annotations

from nlp_sql.query_ast import DateRangePredicate, FilterExpression, Predicate, QueryAST
from nlp_sql.result import QueryError
from nlp_sql.schema import SchemaRegistry

SUPPORTED_AGGREGATIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
SUPPORTED_OPERATORS = {
    ">", "<", ">=", "<=", "=", "!=", "BETWEEN", "IN", "LIKE", "NOT LIKE",
    "IS NULL", "IS NOT NULL",
}
MAX_LIMIT = 1000


class ASTValidator:
    def __init__(self, schema: SchemaRegistry, max_limit: int = MAX_LIMIT) -> None:
        self._schema = schema
        self._max_limit = max_limit

    def validate(self, ast: QueryAST) -> QueryError | None:
        if not self._schema.has_table(ast.source):
            return QueryError("UNKNOWN_TABLE", f"Table '{ast.source}' is not configured.")
        for join in ast.joins:
            if join.join_type not in {"INNER", "LEFT", "RIGHT", "FULL"}:
                return QueryError("UNSUPPORTED_JOIN", f"Unsupported join type {join.join_type}.")
            if not self._schema.has_table(join.table):
                return QueryError("UNKNOWN_TABLE", f"Table '{join.table}' is not configured.")
            if not self._schema.has_column(ast.source, join.left_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Join column '{join.left_column}' is not valid for {ast.source}.",
                )
            if not self._schema.has_column(join.table, join.right_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Join column '{join.right_column}' is not valid for {join.table}.",
                )
        if ast.limit is not None and not (1 <= ast.limit <= self._max_limit):
            return QueryError("INVALID_LIMIT", f"Limit must be between 1 and {self._max_limit}.")
        for column in ast.select:
            if column.name != "*" and not self._has_query_column(ast, column.name):
                return QueryError(
                    "UNKNOWN_COLUMN", f"Column '{column.name}' is not valid for {ast.source}."
                )
        for aggregation in ast.aggregations:
            if aggregation.function not in SUPPORTED_AGGREGATIONS:
                return QueryError(
                    "UNSUPPORTED_AGGREGATION", f"Unsupported aggregation {aggregation.function}."
                )
            if aggregation.column != "*" and not self._has_query_column(ast, aggregation.column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Column '{aggregation.column}' is not valid for {ast.source}.",
                )
        for predicate in ast.filters:
            error = self._validate_predicate(ast, predicate)
            if error is not None:
                return error
        if ast.where is not None:
            error = self._validate_filter_expression(ast, ast.where)
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
            if not self._has_query_column(ast, having_predicate.left.column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Column '{having_predicate.left.column}' is not valid for {ast.source}.",
                )
        for group_column in ast.group_by:
            if not self._has_query_column(ast, group_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Group by column '{group_column}' is not valid for {ast.source}.",
                )
        for order in ast.order_by:
            if order.column is not None and not self._has_query_column(ast, order.column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Order by column '{order.column}' is not valid for {ast.source}.",
                )
        if ast.having and not ast.group_by:
            return QueryError("INVALID_HAVING", "HAVING requires GROUP BY in this engine.")
        return None
    def _has_query_column(self, ast: QueryAST, reference: str) -> bool:
        if "." in reference:
            table, column = reference.split(".", 1)
            allowed_tables = {ast.source, *(join.table for join in ast.joins)}
            return table in allowed_tables and self._schema.has_column(table, column)
        return self._schema.has_column(ast.source, reference)

    def _validate_filter_expression(
        self, ast: QueryAST, expression: FilterExpression
    ) -> QueryError | None:
        if expression.kind == "predicate":
            if expression.predicate is None:
                return QueryError("INVALID_FILTER", "Predicate expression is missing a predicate.")
            return self._validate_predicate(ast, expression.predicate)
        if expression.kind == "not":
            if len(expression.children) != 1:
                return QueryError("INVALID_FILTER", "NOT requires exactly one child.")
        elif expression.kind in {"and", "or"}:
            if len(expression.children) < 2:
                return QueryError("INVALID_FILTER", f"{expression.kind.upper()} requires two children.")
        else:
            return QueryError("INVALID_FILTER", f"Unsupported filter expression '{expression.kind}'.")
        for child in expression.children:
            error = self._validate_filter_expression(ast, child)
            if error is not None:
                return error
        return None
    def _validate_predicate(
        self, ast: QueryAST, predicate: Predicate | DateRangePredicate
    ) -> QueryError | None:
        if isinstance(predicate, DateRangePredicate):
            if not self._has_query_column(ast, predicate.column):
                return QueryError(
                    "UNKNOWN_COLUMN", f"Date column '{predicate.column}' is not valid for {ast.source}."
                )
            return None
        if not self._has_query_column(ast, predicate.column):
            return QueryError(
                "UNKNOWN_COLUMN", f"Column '{predicate.column}' is not valid for {ast.source}."
            )
        if predicate.operator not in SUPPORTED_OPERATORS:
            return QueryError("UNSUPPORTED_OPERATOR", f"Unsupported operator {predicate.operator}.")
        return None
