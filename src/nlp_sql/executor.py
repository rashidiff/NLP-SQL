"""Optional database execution layer kept separate from parsing and compilation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from nlp_sql.result import SqlQuery


class SQLiteExecutor:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def execute(self, query: SqlQuery, max_rows: int = 100) -> list[dict[str, Any]]:
        sql = query.sql
        if " LIMIT ?" not in sql.upper() and " LIMIT " not in sql.upper():
            sql = f"{sql} LIMIT ?"
            parameters = (*query.parameters, max_rows)
        else:
            parameters = query.parameters

        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(sql, parameters)
            rows = cursor.fetchmany(max_rows)
        return [dict(row) for row in rows]
