"""Parameterized SQL compiler for validated Query ASTs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from nlp_sql.query_ast import (
    DateRangePredicate,
    CteExpression,
    FilterExpression,
    Predicate,
    QueryAST,
    OrderBy,
    SubqueryExpression,
    WindowExpression,
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

    def quote_identifier(self, identifier: str) -> str:
        return ".".join(
            '"' + part.replace('"', '""') + '"' for part in identifier.split(".")
        )


@dataclass(frozen=True)
class SqlCompiler:
    dialect: SqlDialect = SqlDialect()

    def compile(self, ast: QueryAST) -> SqlQuery:
        parameters: list[Any] = []
        sql = self._compile_statement(ast, parameters)
        return SqlQuery(sql, tuple(parameters))

    def _compile_statement(self, ast: QueryAST, parameters: list[Any]) -> str:
        cte_sqls: list[str] = []
        for cte in ast.ctes:
            cte_sql = self._compile_statement(cte.query, parameters)
            column_list = (
                f" ({', '.join(self._identifier(item) for item in cte.columns)})"
                if cte.columns
                else ""
            )
            cte_sqls.append(f"{self._identifier(cte.name)}{column_list} AS ({cte_sql})")
        sql = [
            "SELECT",
            "DISTINCT" if ast.distinct else "",
            self._compile_select(ast),
            "FROM",
            self._identifier(ast.source),
        ]
        for join in ast.joins:
            sql.extend(
                [
                    join.join_type,
                    "JOIN",
                    self._identifier(join.table),
                    "ON",
                    f"{self._identifier(join.left_table or ast.source)}."
                    f"{self._identifier(join.left_column)} = "
                    f"{self._identifier(join.table)}.{self._identifier(join.right_column)}",
                ]
            )
        where_sql = self._compile_where(ast, parameters)
        if where_sql:
            sql.extend(["WHERE", where_sql])
        if ast.group_by:
            sql.extend(["GROUP BY", ", ".join(self._identifier(item) for item in ast.group_by)])
        having_sql = self._compile_having(ast, parameters)
        if having_sql:
            sql.extend(["HAVING", having_sql])
        if ast.order_by:
            sql.extend(["ORDER BY", self._compile_order_by(ast)])
        if ast.limit is not None:
            sql.extend(["LIMIT", self._placeholder(parameters)])
            parameters.append(ast.limit)
        statement = " ".join(part for part in sql if part)
        if cte_sqls:
            statement = "WITH " + ", ".join(cte_sqls) + " " + statement
        return statement

    def _compile_select(self, ast: QueryAST) -> str:
        parts: list[str] = []
        for column in ast.select:
            if isinstance(column, WindowExpression):
                parts.append(self._compile_window(column))
            else:
                parts.append(self._identifier(column.name))
        for aggregation in ast.aggregations:
            distinct = "DISTINCT " if aggregation.distinct else ""
            expression = f"{aggregation.function}({distinct}{self._identifier(aggregation.column)})"
            if aggregation.alias:
                expression += f" AS {self._identifier(aggregation.alias)}"
            parts.append(expression)
        return ", ".join(parts) if parts else "*"

    def _compile_window(self, expression: WindowExpression) -> str:
        argument = "*" if expression.column == "*" else self._identifier(expression.column)
        parts = [f"{expression.function}({argument})", "OVER ("]
        if expression.partition_by:
            parts.extend(["PARTITION BY", ", ".join(self._identifier(item) for item in expression.partition_by)])
        if expression.order_by:
            if expression.partition_by:
                parts.append("")
            parts.extend(["ORDER BY", self._compile_order_items(expression.order_by)])
        parts.append(")")
        compiled = " ".join(part for part in parts if part)
        return f"{compiled} AS {self._identifier(expression.alias)}" if expression.alias else compiled

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
                clauses.append(f"{self._identifier(predicate.column)} >= {self._placeholder(parameters)}")
                parameters.append(predicate.start.isoformat())
            if predicate.end != date.max:
                clauses.append(f"{self._identifier(predicate.column)} < {self._placeholder(parameters)}")
                parameters.append(predicate.end.isoformat())
            return " AND ".join(clauses) or "1 = 1"
        if predicate.operator == "BETWEEN":
            start, end = predicate.value
            first = self._placeholder(parameters)
            second = self._placeholder(parameters, offset=1)
            parameters.extend([start, end])
            return (
                f"{self._identifier(predicate.column)} BETWEEN {first} "
                f"AND {second}"
            )
        if predicate.operator in {"IS NULL", "IS NOT NULL"}:
            return f"{self._identifier(predicate.column)} {predicate.operator}"
        if predicate.operator == "IN":
            if isinstance(predicate.value, SubqueryExpression):
                subquery = self._compile_statement(predicate.value.query, parameters)
                return f"{self._identifier(predicate.column)} IN ({subquery})"
            values = predicate.value if isinstance(predicate.value, (list, tuple)) else ()
            if not values:
                raise ValueError("IN requires at least one value")
            placeholders = ", ".join(
                self._placeholder(parameters, offset=index) for index in range(len(values))
            )
            parameters.extend(values)
            return f"{self._identifier(predicate.column)} IN ({placeholders})"
        if predicate.operator == "NOT IN" and isinstance(predicate.value, SubqueryExpression):
            subquery = self._compile_statement(predicate.value.query, parameters)
            return f"{self._identifier(predicate.column)} NOT IN ({subquery})"
        if predicate.operator == "EXISTS" and isinstance(predicate.value, SubqueryExpression):
            subquery = self._compile_statement(predicate.value.query, parameters)
            return f"EXISTS ({subquery})"
        if (
            isinstance(predicate.value, str)
            and predicate.operator in {"=", "!="}
            and "address" in predicate.column
        ):
            placeholder = self._placeholder(parameters)
            parameters.append(predicate.value)
            return (
                f"LOWER(REPLACE(REPLACE({self._identifier(predicate.column)}, ',', ''), '  ', ' ')) "
                f"{predicate.operator} LOWER({placeholder})"
            )
        if predicate.operator in {"=", "!=", "LIKE", "NOT LIKE"} and isinstance(predicate.value, str):
            placeholder = self._placeholder(parameters)
            parameters.append(predicate.value)
            return (
                f"LOWER({self._identifier(predicate.column)}) {predicate.operator} "
                f"LOWER({placeholder})"
            )
        placeholder = self._placeholder(parameters)
        parameters.append(predicate.value)
        return f"{self._identifier(predicate.column)} {predicate.operator} {placeholder}"

    def _compile_having(self, ast: QueryAST, parameters: list[Any]) -> str:
        clauses: list[str] = []
        for predicate in ast.having:
            distinct = "DISTINCT " if predicate.left.distinct else ""
            left = f"{predicate.left.function}({distinct}{self._identifier(predicate.left.column)})"
            clauses.append(f"{left} {predicate.operator} {self._placeholder(parameters)}")
            parameters.append(predicate.value)
        return " AND ".join(clauses)

    def _placeholder(self, parameters: list[Any], *, offset: int = 0) -> str:
        return self.dialect.placeholder_for(len(parameters) + offset + 1)

    def _identifier(self, identifier: str) -> str:
        return self.dialect.quote_identifier(identifier)

    def _compile_order_by(self, ast: QueryAST) -> str:
        return self._compile_order_items(ast.order_by)

    def _compile_order_items(self, orders: tuple[OrderBy, ...]) -> str:
        parts: list[str] = []
        for order in orders:
            if order.aggregation is not None:
                distinct = "DISTINCT " if order.aggregation.distinct else ""
                expression = f"{order.aggregation.function}({distinct}{self._identifier(order.aggregation.column)})"
            else:
                expression = self._identifier(order.column) if order.column else ""
            parts.append(f"{expression} {order.direction.value}")
        return ", ".join(parts)
