"""Transparent metadata graph for expanding schema retrieval candidates."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

from nlp_sql.dataset_registry import DatasetRegistry
from nlp_sql.retrieval.base import CandidateKind, SchemaDocument, documents_from_schema


class SchemaEdgeKind(StrEnum):
    BELONGS_TO = "belongs_to"
    FOREIGN_KEY = "foreign_key"
    SEMANTIC_ALIAS = "semantic_alias"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"


@dataclass(frozen=True)
class SchemaNode:
    key: str
    kind: CandidateKind
    label: str
    dataset_id: str
    table: str | None = None
    column: str | None = None


@dataclass(frozen=True)
class SchemaEdge:
    source: str
    target: str
    kind: SchemaEdgeKind
    weight: float = 1.0


class SchemaGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, SchemaNode] = {}
        self._edges: dict[str, list[SchemaEdge]] = defaultdict(list)

    @classmethod
    def from_registry(cls, registry: DatasetRegistry) -> SchemaGraph:
        graph = cls()
        for dataset in registry.list_datasets():
            for document in documents_from_schema(
                dataset.schema,
                dataset_id=dataset.id,
                dataset_title=dataset.title,
                dataset_description=dataset.description,
            ):
                graph.add_document(document)
        graph._infer_lightweight_edges()
        return graph

    def add_document(self, document: SchemaDocument) -> None:
        self._nodes[document.key] = SchemaNode(
            key=document.key,
            kind=document.kind,
            label=document.human_name,
            dataset_id=document.dataset_id,
            table=document.table,
            column=document.column,
        )
        if document.kind == CandidateKind.COLUMN and document.table is not None:
            table_key = "/".join((document.dataset_id, CandidateKind.TABLE.value, document.table, "-"))
            self.add_edge(document.key, table_key, SchemaEdgeKind.BELONGS_TO)
            for alias in document.aliases:
                alias_key = "/".join(
                    (document.dataset_id, CandidateKind.CONCEPT.value, document.table, alias)
                )
                self._nodes.setdefault(
                    alias_key,
                    SchemaNode(
                        key=alias_key,
                        kind=CandidateKind.CONCEPT,
                        label=alias,
                        dataset_id=document.dataset_id,
                        table=document.table,
                    ),
                )
                self.add_edge(alias_key, document.key, SchemaEdgeKind.SEMANTIC_ALIAS, 0.8)

    def add_edge(
        self,
        source: str,
        target: str,
        kind: SchemaEdgeKind,
        weight: float = 1.0,
    ) -> None:
        self._edges[source].append(SchemaEdge(source, target, kind, weight))
        self._edges[target].append(SchemaEdge(target, source, kind, weight))

    def expand(
        self,
        seed_keys: tuple[str, ...],
        *,
        max_depth: int = 2,
        edge_kinds: tuple[SchemaEdgeKind, ...] | None = None,
    ) -> tuple[SchemaNode, ...]:
        allowed = set(edge_kinds) if edge_kinds is not None else None
        seen = set(seed_keys)
        queue: deque[tuple[str, int]] = deque((key, 0) for key in seed_keys)
        while queue:
            key, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._edges.get(key, ()):
                if allowed is not None and edge.kind not in allowed:
                    continue
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                queue.append((edge.target, depth + 1))
        return tuple(self._nodes[key] for key in seen if key in self._nodes)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.__dict__ for node in self._nodes.values()],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind.value,
                    "weight": edge.weight,
                }
                for edges in self._edges.values()
                for edge in edges
                if edge.source < edge.target
            ],
        }

    def _infer_lightweight_edges(self) -> None:
        columns_by_name: dict[str, list[str]] = defaultdict(list)
        for key, node in self._nodes.items():
            if node.kind == CandidateKind.COLUMN and node.column:
                columns_by_name[node.column].append(key)
                if node.column.endswith("_id"):
                    target_name = node.column.removesuffix("_id")
                    columns_by_name[target_name].append(key)
        for keys in columns_by_name.values():
            if len(keys) < 2:
                continue
            for index, source in enumerate(keys):
                for target in keys[index + 1 :]:
                    self.add_edge(source, target, SchemaEdgeKind.SIMILAR_TO, 0.5)
