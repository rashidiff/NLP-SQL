"""AST validation before SQL compilation."""

from __future__ import annotations

import re

from nlp_sql.query_ast import (
    CteExpression,
    DateRangePredicate,
    FilterExpression,
    Predicate,
    QueryAST,
    SubqueryExpression,
    WindowExpression,
)
from nlp_sql.result import QueryError
from nlp_sql.schema import SchemaRegistry

SUPPORTED_AGGREGATIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
SUPPORTED_OPERATORS = {
    ">", "<", ">=", "<=", "=", "!=", "BETWEEN", "IN", "LIKE", "NOT LIKE",
    "IS NULL", "IS NOT NULL", "NOT IN", "EXISTS",
}
MAX_LIMIT = 1000


class ASTValidator:
    def __init__(self, schema: SchemaRegistry, max_limit: int = MAX_LIMIT) -> None:
        self._schema = schema
        self._max_limit = max_limit

    def validate(self, ast: QueryAST) -> QueryError | None:
        for cte in ast.ctes:
            if not _valid_identifier(cte.name):
                return QueryError("INVALID_IDENTIFIER", f"Invalid CTE name '{cte.name}'.")
            if any(not _valid_identifier(column) for column in cte.columns):
                return QueryError("INVALID_IDENTIFIER", "CTE columns must be valid identifiers.")
        cte_columns = self._cte_columns(ast)
        if not self._schema.has_table(ast.source) and ast.source not in cte_columns:
            return QueryError("UNKNOWN_TABLE", f"Table '{ast.source}' is not configured.")
        for cte in ast.ctes:
            error = self.validate(cte.query)
            if error is not None:
                return error
        for join in ast.joins:
            if join.join_type not in {"INNER", "LEFT", "RIGHT", "FULL"}:
                return QueryError("UNSUPPORTED_JOIN", f"Unsupported join type {join.join_type}.")
            if not self._schema.has_table(join.table):
                return QueryError("UNKNOWN_TABLE", f"Table '{join.table}' is not configured.")
            available_tables = {ast.source, *(item.table for item in ast.joins)}
            left_table = join.left_table or ast.source
            if left_table not in available_tables:
                return QueryError("UNKNOWN_TABLE", f"Join table '{left_table}' is not configured.")
            if not self._schema.has_column(left_table, join.left_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Join column '{join.left_column}' is not valid for {left_table}.",
                )
            if not self._schema.has_column(join.table, join.right_column):
                return QueryError(
                    "UNKNOWN_COLUMN",
                    f"Join column '{join.right_column}' is not valid for {join.table}.",
                )
        if ast.limit is not None and not (1 <= ast.limit <= self._max_limit):
            return QueryError("INVALID_LIMIT", f"Limit must be between 1 and {self._max_limit}.")
        for column in ast.select:
            if isinstance(column, WindowExpression):
                if column.function not in {
                    "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "SUM", "AVG", "MIN", "MAX", "COUNT"
                }:
                    return QueryError("UNSUPPORTED_WINDOW", f"Unsupported window function {column.function}.")
                if column.column != "*" and not self._has_query_column(ast, column.column):
                    return QueryError("UNKNOWN_COLUMN", f"Column '{column.column}' is not valid for {ast.source}.")
                for partition_column in column.partition_by:
                    if not self._has_query_column(ast, partition_column):
                        return QueryError("UNKNOWN_COLUMN", f"Column '{partition_column}' is not valid for {ast.source}.")
                for window_order in column.order_by:
                    if window_order.column and not self._has_query_column(ast, window_order.column):
                        return QueryError("UNKNOWN_COLUMN", f"Column '{window_order.column}' is not valid for {ast.source}.")
                if column.alias is not None and not _valid_identifier(column.alias):
                    return QueryError("INVALID_IDENTIFIER", f"Invalid window alias '{column.alias}'.")
                continue
            if column.name != "*" and not self._has_query_column(ast, column.name):
                return QueryError(
                    "UNKNOWN_COLUMN", f"Column '{column.name}' is not valid for {ast.source}."
                )
        for aggregation in ast.aggregations:
            if aggregation.function not in SUPPORTED_AGGREGATIONS:
                return QueryError(
                    "UNSUPPORTED_AGGREGATION", f"Unsupported aggregation {aggregation.function}."
                )
            if aggregation.alias is not None and not _valid_identifier(aggregation.alias):
                return QueryError("INVALID_IDENTIFIER", f"Invalid aggregation alias '{aggregation.alias}'.")
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
            return table in allowed_tables and self._has_column(ast, table, column)
        if ast.source in self._cte_columns(ast):
            return reference in self._cte_columns(ast)[ast.source]
        return self._schema.has_column(ast.source, reference)

    def _has_column(self, ast: QueryAST, table: str, column: str) -> bool:
        if table in self._cte_columns(ast):
            return column in self._cte_columns(ast)[table]
        return self._schema.has_column(table, column)

    def _cte_columns(self, ast: QueryAST) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for cte in ast.ctes:
            columns = set(cte.columns)
            if not columns:
                columns = {
                    item.name for item in cte.query.select if hasattr(item, "name")
                }
                columns.update(
                    item.alias
                    for item in cte.query.select
                    if getattr(item, "alias", None)
                )
            result[cte.name] = columns
        return result

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
        if predicate.operator == "EXISTS":
            if not isinstance(predicate.value, SubqueryExpression):
                return QueryError("INVALID_SUBQUERY", "EXISTS requires a subquery.")
            return self.validate(predicate.value.query)
        if predicate.operator in {"IN", "NOT IN"} and isinstance(
            predicate.value, SubqueryExpression
        ):
            return self.validate(predicate.value.query)
        if not self._has_query_column(ast, predicate.column):
            return QueryError(
                "UNKNOWN_COLUMN", f"Column '{predicate.column}' is not valid for {ast.source}."
            )
        if predicate.operator not in SUPPORTED_OPERATORS:
            return QueryError("UNSUPPORTED_OPERATOR", f"Unsupported operator {predicate.operator}.")
        value_error = self._validate_value(ast, predicate)
        if value_error is not None:
            return value_error
        return None

    def _validate_value(self, ast: QueryAST, predicate: Predicate) -> QueryError | None:
        if predicate.operator in {"IS NULL", "IS NOT NULL", "EXISTS"}:
            return None
        column = predicate.column.split(".", 1)[-1]
        table = predicate.column.split(".", 1)[0] if "." in predicate.column else ast.source
        metadata = self._schema.get_column(table, column)
        if metadata is None:
            return None
        values: object = predicate.value
        if predicate.operator == "BETWEEN":
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                return QueryError("INVALID_VALUE", "BETWEEN requires exactly two values.")
            values = tuple(values)
        elif predicate.operator in {"IN", "NOT IN"}:
            if not isinstance(values, (tuple, list)) or not values:
                return QueryError("INVALID_VALUE", "IN requires a non-empty value list.")
        else:
            values = (values,)
        for value in values:
            if metadata.type in {"integer", "decimal"} and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                return QueryError(
                    "INVALID_VALUE",
                    f"Column '{predicate.column}' requires numeric values.",
                )
            if metadata.type == "datetime" and not isinstance(value, str):
                return QueryError(
                    "INVALID_VALUE",
                    f"Column '{predicate.column}' requires date strings.",
                )
        return None


def _valid_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))
