"""Optional database execution layer kept separate from parsing and compilation."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlp_sql.result import SqlQuery


@dataclass(frozen=True)
class ExecutionPolicy:
    max_rows: int = 100
    timeout_ms: int = 5_000
    progress_interval: int = 1_000


class SQLiteExecutor:
    def __init__(self, db_path: Path, policy: ExecutionPolicy | None = None) -> None:
        self._db_path = db_path
        self._policy = policy or ExecutionPolicy()

    def execute(self, query: SqlQuery, max_rows: int = 100) -> list[dict[str, Any]]:
        row_limit = min(max_rows, self._policy.max_rows)
        sql = query.sql
        if " LIMIT ?" not in sql.upper() and " LIMIT " not in sql.upper():
            sql = f"{sql} LIMIT ?"
            parameters = (*query.parameters, row_limit)
        else:
            parameters = query.parameters

        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            started = time.monotonic()

            def progress() -> int:
                elapsed_ms = (time.monotonic() - started) * 1000
                return int(elapsed_ms >= self._policy.timeout_ms)

            connection.set_progress_handler(progress, self._policy.progress_interval)
            cursor = connection.execute(sql, parameters)
            rows = cursor.fetchmany(row_limit)
        return [dict(row) for row in rows]

    def explain(self, query: SqlQuery) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(f"EXPLAIN QUERY PLAN {query.sql}", query.parameters).fetchall()
        return [dict(row) for row in rows]
