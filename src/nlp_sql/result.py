"""Structured pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp_sql.query_ast import QueryAST


@dataclass(frozen=True)
class QueryError:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class SqlQuery:
    sql: str
    parameters: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"sql": self.sql, "parameters": list(self.parameters)}


@dataclass(frozen=True)
class PipelineResult:
    success: bool
    ast: QueryAST | None = None
    sql: SqlQuery | None = None
    error: QueryError | None = None
    requires_clarification: bool = False
    interpretation: dict[str, Any] | None = None
    matched_rules: tuple[str, ...] = ()
    rows: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"success": self.success}
        if self.ast is not None:
            payload["ast"] = self.ast.to_dict()
        if self.sql is not None:
            payload.update(self.sql.to_dict())
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        if self.requires_clarification:
            payload["requires_clarification"] = True
        if self.interpretation is not None:
            payload["interpretation"] = self.interpretation
        if self.matched_rules:
            payload["matched_rules"] = list(self.matched_rules)
        if self.rows is not None:
            payload["rows"] = self.rows
        return payload
