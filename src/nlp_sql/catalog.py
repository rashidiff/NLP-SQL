"""Database profiling utilities for schema-aware natural-language planning."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlp_sql.schema import SchemaRegistry


@dataclass(frozen=True)
class ColumnProfile:
    table: str
    column: str
    row_count: int
    non_null_count: int
    distinct_count: int
    minimum: Any = None
    maximum: Any = None
    sample_values: tuple[str, ...] = ()

    @property
    def null_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return (self.row_count - self.non_null_count) / self.row_count

    def metadata(self) -> dict[str, str]:
        values = ", ".join(self.sample_values)
        return {
            "row_count": str(self.row_count),
            "non_null_count": str(self.non_null_count),
            "distinct_count": str(self.distinct_count),
            "null_ratio": f"{self.null_ratio:.4f}",
            "minimum": "" if self.minimum is None else str(self.minimum),
            "maximum": "" if self.maximum is None else str(self.maximum),
            "sample_values": values,
        }


def profile_sqlite(
    db_path: Path,
    schema: SchemaRegistry,
    *,
    max_sample_values: int = 12,
) -> dict[str, ColumnProfile]:
    """Profile whitelisted SQLite columns without interpolating user identifiers."""

    profiles: dict[str, ColumnProfile] = {}
    with sqlite3.connect(db_path) as connection:
        for table in schema.tables.values():
            quoted_table = _quote_identifier(table.name)
            row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0])
            for column in table.columns.values():
                quoted_column = _quote_identifier(column.name)
                row = connection.execute(
                    "SELECT COUNT(*), COUNT({0}), COUNT(DISTINCT {0}), MIN({0}), MAX({0}) "
                    "FROM {1}".format(quoted_column, quoted_table)
                ).fetchone()
                samples = connection.execute(
                    "SELECT DISTINCT {0} FROM {1} WHERE {0} IS NOT NULL LIMIT ?".format(
                        quoted_column, quoted_table
                    ),
                    (max_sample_values,),
                ).fetchall()
                profiles[f"{table.name}.{column.name}"] = ColumnProfile(
                    table=table.name,
                    column=column.name,
                    row_count=row_count,
                    non_null_count=int(row[1]),
                    distinct_count=int(row[2]),
                    minimum=row[3],
                    maximum=row[4],
                    sample_values=tuple(str(item[0]) for item in samples),
                )
    return profiles


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
