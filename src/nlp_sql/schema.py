"""Schema registry and alias resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
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
