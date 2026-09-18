"""Schema registry and alias resolution."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    name: str
    aliases: tuple[str, ...]
    columns: dict[str, Column]


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str
    type: str


@dataclass(frozen=True)
class Resolution:
    canonical: str
    matched_alias: str


class SchemaRegistry:
    """Whitelisted tables and columns with deterministic alias lookup."""

    def __init__(self, tables: dict[str, Table]) -> None:
        self._tables = tables
        self._table_aliases = self._build_table_aliases(tables)
        self._column_aliases = self._build_column_aliases(tables)

    @classmethod
    def from_dict(cls, raw_schema: dict[str, Any]) -> SchemaRegistry:
        tables: dict[str, Table] = {}
        for table_name, table_data in raw_schema.items():
            columns: dict[str, Column] = {}
            for column_name, column_data in table_data.get("columns", {}).items():
                aliases = tuple(column_data.get("aliases", (column_name,)))
                columns[column_name] = Column(
                    name=column_name,
                    type=column_data["type"],
                    aliases=aliases,
                )
            tables[table_name] = Table(
                name=table_name,
                aliases=tuple(table_data.get("aliases", (table_name,))),
                columns=columns,
            )
        return cls(tables)

    @classmethod
    def default(cls) -> SchemaRegistry:
        schema_file = resources.files("nlp_sql.config").joinpath("default_schema.json")
        with schema_file.open(encoding="utf-8") as file:
            raw_schema = json.load(file)
        return cls.from_dict(raw_schema)

    @classmethod
    def from_sqlite(
        cls,
        db_path: Path,
        table_aliases: dict[str, tuple[str, ...]] | None = None,
        column_aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> SchemaRegistry:
        table_aliases = table_aliases or {}
        column_aliases = column_aliases or {}
        tables: dict[str, Table] = {}
        with sqlite3.connect(db_path) as connection:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (table_name,) in table_rows:
                columns: dict[str, Column] = {}
                for _, column_name, sqlite_type, *_ in connection.execute(
                    f"PRAGMA table_info({table_name})"
                ):
                    aliases = _column_aliases(column_name, column_aliases.get(column_name, ()))
                    columns[column_name] = Column(
                        name=column_name,
                        type=_map_sqlite_type(column_name, sqlite_type),
                        aliases=aliases,
                    )
                tables[table_name] = Table(
                    name=table_name,
                    aliases=_table_aliases(table_name, table_aliases.get(table_name, ())),
                    columns=columns,
                )
        return cls(tables)

    @property
    def tables(self) -> dict[str, Table]:
        return dict(self._tables)

    def has_table(self, table: str) -> bool:
        return table in self._tables

    def has_column(self, table: str, column: str) -> bool:
        return table in self._tables and column in self._tables[table].columns

    def get_column(self, table: str, column: str) -> Column | None:
        if not self.has_column(table, column):
            return None
        return self._tables[table].columns[column]

    def resolve_table(self, phrase: str) -> Resolution | None:
        alias = phrase.strip().lower()
        canonical = self._table_aliases.get(alias)
        if canonical is None:
            return None
        return Resolution(canonical=canonical, matched_alias=alias)

    def resolve_column(self, phrase: str, table: str | None = None) -> tuple[ColumnRef, ...]:
        alias = phrase.strip().lower()
        candidates = self._column_aliases.get(alias, ())
        if table is not None:
            candidates = tuple(candidate for candidate in candidates if candidate.table == table)
        return candidates

    def _build_table_aliases(self, tables: dict[str, Table]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for table in tables.values():
            aliases[table.name] = table.name
            for alias in table.aliases:
                aliases[alias.lower()] = table.name
        return aliases

    def _build_column_aliases(self, tables: dict[str, Table]) -> dict[str, tuple[ColumnRef, ...]]:
        aliases: dict[str, list[ColumnRef]] = {}
        for table in tables.values():
            for column in table.columns.values():
                ref = ColumnRef(table=table.name, column=column.name, type=column.type)
                for alias in (column.name, *column.aliases):
                    aliases.setdefault(alias.lower(), []).append(ref)
        return {alias: tuple(refs) for alias, refs in aliases.items()}


def _table_aliases(table_name: str, configured: tuple[str, ...]) -> tuple[str, ...]:
    readable = table_name.replace("_", " ")
    return tuple(dict.fromkeys((table_name, readable, *configured)))


def _column_aliases(column_name: str, configured: tuple[str, ...]) -> tuple[str, ...]:
    readable = column_name.replace("_", " ")
    return tuple(dict.fromkeys((column_name, readable, *configured)))


def _map_sqlite_type(column_name: str, sqlite_type: str) -> str:
    if column_name.endswith("_date") or column_name in {"date", "created_at", "sale_date"}:
        return "datetime"
    normalized = sqlite_type.upper()
    if "INT" in normalized:
        return "integer"
    if "REAL" in normalized or "FLOA" in normalized or "DOUB" in normalized or "NUM" in normalized:
        return "decimal"
    return "string"
