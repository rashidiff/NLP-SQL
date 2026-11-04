"""Local query memory for semantic ranking feedback."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryMemoryEntry:
    query: str
    plan_signature: str
    success: bool
    score: float
    error_code: str | None = None


class QueryMemory:
    """Small SQLite journal used as a local, privacy-preserving ranking signal."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    plan_signature TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    score REAL NOT NULL,
                    error_code TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.commit()

    def record(self, entry: QueryMemoryEntry) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO query_memory
                (query, plan_signature, success, score, error_code, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.query,
                    entry.plan_signature,
                    int(entry.success),
                    entry.score,
                    entry.error_code,
                    time.time(),
                ),
            )
            connection.commit()

    def success_prior(self, query: str, plan_signature: str) -> float:
        with sqlite3.connect(self._path) as connection:
            rows = connection.execute(
                "SELECT success, score FROM query_memory WHERE query = ? AND plan_signature = ? "
                "ORDER BY id DESC LIMIT 20",
                (query, plan_signature),
            ).fetchall()
        if not rows:
            return 0.0
        return sum((1.0 if success else -1.0) * float(score) for success, score in rows) / len(rows)

    def recent(self, limit: int = 50) -> tuple[dict[str, object], ...]:
        with sqlite3.connect(self._path) as connection:
            rows = connection.execute(
                "SELECT query, plan_signature, success, score, error_code, created_at "
                "FROM query_memory ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(
            {
                "query": query,
                "plan_signature": signature,
                "success": bool(success),
                "score": score,
                "error_code": error_code,
                "created_at": created_at,
            }
            for query, signature, success, score, error_code, created_at in rows
        )


def plan_signature(plan: object) -> str:
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()
    return json.dumps(plan, sort_keys=True, default=str)
