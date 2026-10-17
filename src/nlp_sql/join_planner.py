"""Foreign-key-based join path planning for relational query plans."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nlp_sql.query_ast import JoinExpression
from nlp_sql.schema import SchemaRegistry


@dataclass(frozen=True)
class JoinPath:
    source: str
    target: str
    joins: tuple[JoinExpression, ...]


class JoinPlanner:
    def __init__(self, schema: SchemaRegistry) -> None:
        self._schema = schema

    def find_path(self, source: str, target: str) -> JoinPath | None:
        if source == target:
            return JoinPath(source, target, ())
        queue: deque[tuple[str, tuple[JoinExpression, ...]]] = deque([(source, ())])
        visited = {source}
        while queue:
            current, joins = queue.popleft()
            for next_table, join in self._neighbors(current):
                if next_table in visited:
                    continue
                next_joins = joins + (join,)
                if next_table == target:
                    return JoinPath(source, target, next_joins)
                visited.add(next_table)
                queue.append((next_table, next_joins))
        return None

    def _neighbors(self, table: str) -> tuple[tuple[str, JoinExpression], ...]:
        neighbors: list[tuple[str, JoinExpression]] = []
        for foreign_key in self._schema.foreign_keys():
            if foreign_key.table == table:
                neighbors.append(
                    (
                        foreign_key.referenced_table,
                        JoinExpression(
                            table=foreign_key.referenced_table,
                            left_table=table,
                            left_column=foreign_key.column,
                            right_column=foreign_key.referenced_column,
                        ),
                    )
                )
            elif foreign_key.referenced_table == table:
                neighbors.append(
                    (
                        foreign_key.table,
                        JoinExpression(
                            table=foreign_key.table,
                            left_table=table,
                            left_column=foreign_key.referenced_column,
                            right_column=foreign_key.column,
                        ),
                    )
                )
        return tuple(neighbors)
