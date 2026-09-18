"""Security validation for natural-language inputs and generated SQL."""

from __future__ import annotations

import re

from nlp_sql.result import QueryError, SqlQuery

FORBIDDEN_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
}


class SafetyValidator:
    def validate_input(self, text: str) -> QueryError | None:
        tokens = set(re.findall(r"[a-z_]+", text.lower()))
        forbidden = sorted(tokens & FORBIDDEN_SQL_KEYWORDS)
        if forbidden:
            return QueryError(
                "FORBIDDEN_OPERATION", f"Read-only queries only; rejected '{forbidden[0]}'."
            )
        return None

    def validate_sql(self, query: SqlQuery) -> QueryError | None:
        tokens = set(re.findall(r"[a-z_]+", query.sql.lower()))
        forbidden = sorted(tokens & FORBIDDEN_SQL_KEYWORDS)
        if forbidden:
            return QueryError(
                "UNSAFE_SQL", f"Generated SQL contains forbidden keyword '{forbidden[0]}'."
            )
        if ";" in query.sql:
            return QueryError(
                "UNSAFE_SQL", "Generated SQL must contain a single statement without semicolons."
            )
        return None
