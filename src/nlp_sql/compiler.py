"""Parameterized SQL compiler for validated Query ASTs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from nlp_sql.query_ast import DateRangePredicate, QueryAST
from nlp_sql.result import SqlQuery


class SqlDialect:
    placeholder = "?"

    def quote_identifier(self, identifier: str) -> str:
        return identifier


class PostgresDialect(SqlDialect):
    """PostgreSQL-compatible compiler settings using DB-API style placeholders."""


@dataclass(frozen=True)
class SqlCompiler:
    dialect: SqlDialect = PostgresDialect()

    def compile(self, ast: QueryAST) -> SqlQuery:
        parameters: list[Any] = []
        sql = [
            "SELECT",
            self._compile_select(ast),
            "FROM",
            self.dialect.quote_identifier(ast.source),
        ]
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
            sql.extend(["LIMIT", self.dialect.placeholder])
            parameters.append(ast.limit)
        return SqlQuery(" ".join(sql), tuple(parameters))

    def _compile_select(self, ast: QueryAST) -> str:
        parts: list[str] = []
        parts.extend(column.name for column in ast.select)
        for aggregation in ast.aggregations:
            expression = f"{aggregation.function}({aggregation.column})"
            if aggregation.alias:
                expression += f" AS {aggregation.alias}"
            parts.append(expression)
        return ", ".join(parts) if parts else "*"

    def _compile_where(self, ast: QueryAST, parameters: list[Any]) -> str:
        clauses: list[str] = []
        for predicate in ast.filters:
            if isinstance(predicate, DateRangePredicate):
                if predicate.start != date.min:
                    clauses.append(f"{predicate.column} >= {self.dialect.placeholder}")
                    parameters.append(predicate.start.isoformat())
                if predicate.end != date.max:
                    clauses.append(f"{predicate.column} < {self.dialect.placeholder}")
                    parameters.append(predicate.end.isoformat())
                continue
            if predicate.operator == "BETWEEN":
                start, end = predicate.value
                clauses.append(
                    f"{predicate.column} BETWEEN {self.dialect.placeholder} "
                    f"AND {self.dialect.placeholder}"
                )
                parameters.extend([start, end])
                continue
            if isinstance(predicate.value, str) and predicate.operator in {"=", "!="}:
                clauses.append(
                    f"LOWER({predicate.column}) {predicate.operator} "
                    f"LOWER({self.dialect.placeholder})"
                )
                parameters.append(predicate.value)
                continue
            clauses.append(f"{predicate.column} {predicate.operator} {self.dialect.placeholder}")
            parameters.append(predicate.value)
        return " AND ".join(clauses)

    def _compile_having(self, ast: QueryAST, parameters: list[Any]) -> str:
        clauses: list[str] = []
        for predicate in ast.having:
            left = f"{predicate.left.function}({predicate.left.column})"
            clauses.append(f"{left} {predicate.operator} {self.dialect.placeholder}")
            parameters.append(predicate.value)
        return " AND ".join(clauses)

    def _compile_order_by(self, ast: QueryAST) -> str:
        parts: list[str] = []
        for order in ast.order_by:
            if order.aggregation is not None:
                expression = f"{order.aggregation.function}({order.aggregation.column})"
            else:
                expression = order.column or ""
            parts.append(f"{expression} {order.direction.value}")
        return ", ".join(parts)
