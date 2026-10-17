"""Parameterized SQL compiler for validated Query ASTs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from nlp_sql.query_ast import (
    DateRangePredicate,
    FilterExpression,
    Predicate,
    QueryAST,
    SubqueryExpression,
)
from nlp_sql.result import SqlQuery


class SqlDialect:
    placeholder = "?"

    def quote_identifier(self, identifier: str) -> str:
        return identifier

    def placeholder_for(self, parameter_index: int) -> str:
        return self.placeholder


class PostgresDialect(SqlDialect):
    """PostgreSQL compiler settings with numbered placeholders."""

    placeholder = "$1"

    def placeholder_for(self, parameter_index: int) -> str:
        return f"${parameter_index}"


@dataclass(frozen=True)
class SqlCompiler:
    dialect: SqlDialect = SqlDialect()

    def compile(self, ast: QueryAST) -> SqlQuery:
        parameters: list[Any] = []
        sql = self._compile_statement(ast, parameters)
        return SqlQuery(sql, tuple(parameters))

    def _compile_statement(self, ast: QueryAST, parameters: list[Any]) -> str:
        sql = [
            "SELECT",
            "DISTINCT" if ast.distinct else "",
            self._compile_select(ast),
            "FROM",
            self.dialect.quote_identifier(ast.source),
        ]
        for join in ast.joins:
            sql.extend(
                [
                    join.join_type,
                    "JOIN",
                    self.dialect.quote_identifier(join.table),
                    "ON",
                    f"{join.left_table or ast.source}.{join.left_column} = "
                    f"{join.table}.{join.right_column}",
                ]
            )
        where_sql = self._compile_where(ast, parameters)
        if where_sql:
            sql.extend(["WHERE", where_sql])
        if ast.group_by:
            sql.extend(["GROUP BY", ", ".join(ast.group_by)])
        having_sql = self._compile_having(ast, parameters)
        if having_sql:
            sql.extend(["HAVING", having_sql])
        if ast.order_by:
            sql.extend(["ORDER BY", self._compile_order_by(ast)])
        if ast.limit is not None:
            sql.extend(["LIMIT", self._placeholder(parameters)])
            parameters.append(ast.limit)
        return " ".join(part for part in sql if part)

    def _compile_select(self, ast: QueryAST) -> str:
        parts: list[str] = []
        parts.extend(column.name for column in ast.select)
        for aggregation in ast.aggregations:
            distinct = "DISTINCT " if aggregation.distinct else ""
            expression = f"{aggregation.function}({distinct}{aggregation.column})"
            if aggregation.alias:
                expression += f" AS {aggregation.alias}"
            parts.append(expression)
        return ", ".join(parts) if parts else "*"

    def _compile_where(self, ast: QueryAST, parameters: list[Any]) -> str:
        if ast.where is not None:
            return self._compile_filter_expression(ast.where, parameters)
        clauses: list[str] = []
        for predicate in ast.filters:
            clauses.append(self._compile_predicate(predicate, parameters))
        return " AND ".join(clauses)

    def _compile_filter_expression(
        self, expression: FilterExpression, parameters: list[Any]
    ) -> str:
        if expression.kind == "predicate" and expression.predicate is not None:
            return self._compile_predicate(expression.predicate, parameters)
        if expression.kind == "not" and len(expression.children) == 1:
            return f"NOT ({self._compile_filter_expression(expression.children[0], parameters)})"
        if expression.kind in {"and", "or"} and expression.children:
            operator = " AND " if expression.kind == "and" else " OR "
            compiled = [self._compile_filter_expression(child, parameters) for child in expression.children]
            return "(" + operator.join(compiled) + ")"
        raise ValueError(f"Invalid filter expression kind: {expression.kind}")

    def _compile_predicate(
        self, predicate: Predicate | DateRangePredicate, parameters: list[Any]
    ) -> str:
        if isinstance(predicate, DateRangePredicate):
            clauses: list[str] = []
            if predicate.start != date.min:
                clauses.append(f"{predicate.column} >= {self._placeholder(parameters)}")
                parameters.append(predicate.start.isoformat())
            if predicate.end != date.max:
                clauses.append(f"{predicate.column} < {self._placeholder(parameters)}")
                parameters.append(predicate.end.isoformat())
            return " AND ".join(clauses) or "1 = 1"
        if predicate.operator == "BETWEEN":
            start, end = predicate.value
            parameters.extend([start, end])
            return (
                f"{predicate.column} BETWEEN {self._placeholder(parameters)} "
                f"AND {self._placeholder(parameters, offset=1)}"
            )
        if predicate.operator in {"IS NULL", "IS NOT NULL"}:
            return f"{predicate.column} {predicate.operator}"
        if predicate.operator == "IN":
            if isinstance(predicate.value, SubqueryExpression):
                subquery = self._compile_statement(predicate.value.query, parameters)
                return f"{predicate.column} IN ({subquery})"
            values = predicate.value if isinstance(predicate.value, (list, tuple)) else ()
            if not values:
                raise ValueError("IN requires at least one value")
            placeholders = ", ".join(
                self._placeholder(parameters, offset=index) for index in range(len(values))
            )
            parameters.extend(values)
            return f"{predicate.column} IN ({placeholders})"
        if predicate.operator == "NOT IN" and isinstance(predicate.value, SubqueryExpression):
            subquery = self._compile_statement(predicate.value.query, parameters)
            return f"{predicate.column} NOT IN ({subquery})"
        if predicate.operator == "EXISTS" and isinstance(predicate.value, SubqueryExpression):
            subquery = self._compile_statement(predicate.value.query, parameters)
            return f"EXISTS ({subquery})"
        if (
            isinstance(predicate.value, str)
            and predicate.operator in {"=", "!="}
            and "address" in predicate.column
        ):
            parameters.append(predicate.value)
            return (
                f"LOWER(REPLACE(REPLACE({predicate.column}, ',', ''), '  ', ' ')) "
                f"{predicate.operator} LOWER({self._placeholder(parameters)})"
            )
        if predicate.operator in {"=", "!=", "LIKE", "NOT LIKE"} and isinstance(predicate.value, str):
            parameters.append(predicate.value)
            return (
                f"LOWER({predicate.column}) {predicate.operator} "
                f"LOWER({self._placeholder(parameters)})"
            )
        parameters.append(predicate.value)
        return f"{predicate.column} {predicate.operator} {self._placeholder(parameters)}"

    def _compile_having(self, ast: QueryAST, parameters: list[Any]) -> str:
        clauses: list[str] = []
        for predicate in ast.having:
            distinct = "DISTINCT " if predicate.left.distinct else ""
            left = f"{predicate.left.function}({distinct}{predicate.left.column})"
            clauses.append(f"{left} {predicate.operator} {self._placeholder(parameters)}")
            parameters.append(predicate.value)
        return " AND ".join(clauses)

    def _placeholder(self, parameters: list[Any], *, offset: int = 0) -> str:
        return self.dialect.placeholder_for(len(parameters) + offset + 1)

    def _compile_order_by(self, ast: QueryAST) -> str:
        parts: list[str] = []
        for order in ast.order_by:
            if order.aggregation is not None:
                distinct = "DISTINCT " if order.aggregation.distinct else ""
                expression = f"{order.aggregation.function}({distinct}{order.aggregation.column})"
            else:
                expression = order.column or ""
            parts.append(f"{expression} {order.direction.value}")
        return ", ".join(parts)
