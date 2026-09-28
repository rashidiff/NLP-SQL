"""Common schema retrieval contracts.

Retrieval is deliberately separate from SQL generation. These classes can rank
datasets, tables, columns, aliases, descriptions, and values, but they do not
produce SQL and they do not bypass QueryAST validation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from nlp_sql.schema import SchemaRegistry


class CandidateKind(StrEnum):
    DATASET = "dataset"
    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"
    CONCEPT = "concept"


@dataclass(frozen=True)
class SchemaDocument:
    dataset_id: str
    kind: CandidateKind
    table: str | None
    column: str | None
    physical_name: str
    human_name: str
    aliases: tuple[str, ...] = ()
    data_type: str | None = None
    description: str | None = None
    representative_values: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        parts = [self.dataset_id, self.kind.value, self.table or "-", self.column or "-"]
        return "/".join(parts)

    @property
    def text(self) -> str:
        parts = [
            self.dataset_id,
            self.kind.value,
            self.physical_name,
            self.human_name,
            self.data_type or "",
            self.description or "",
            *(self.aliases or ()),
            *(self.representative_values or ()),
        ]
        if self.table:
            parts.append(f"table {self.table}")
        if self.column:
            parts.append(f"column {self.column}")
        return " ".join(part for part in parts if part).lower()

    @property
    def source_hash(self) -> str:
        return stable_hash(self.text)


@dataclass(frozen=True)
class RetrievalScore:
    lexical: float = 0.0
    embedding: float = 0.0
    type_compatibility: float = 0.0
    parser_evidence: float = 0.0
    graph_boost: float = 0.0
    final: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "lexical": self.lexical,
            "embedding": self.embedding,
            "type_compatibility": self.type_compatibility,
            "parser_evidence": self.parser_evidence,
            "graph_boost": self.graph_boost,
            "final": self.final,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    document: SchemaDocument
    score: RetrievalScore
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.document.dataset_id,
            "kind": self.document.kind.value,
            "table": self.document.table,
            "column": self.document.column,
            "physical_name": self.document.physical_name,
            "human_name": self.document.human_name,
            "score": self.score.to_dict(),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_weight: float = 0.45
    embedding_weight: float = 0.45
    type_weight: float = 0.05
    parser_weight: float = 0.05
    graph_weight: float = 0.05
    confidence_threshold: float = 0.30
    top_k: int = 8


class SemanticSchemaRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        parser_evidence: dict[str, float] | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Return ranked schema candidates without generating SQL."""


def documents_from_schema(
    schema: SchemaRegistry,
    *,
    dataset_id: str = "default",
    dataset_title: str | None = None,
    dataset_description: str | None = None,
    representative_values: dict[str, tuple[str, ...]] | None = None,
) -> tuple[SchemaDocument, ...]:
    documents: list[SchemaDocument] = []
    table_values = representative_values or {}
    if dataset_title or dataset_description:
        documents.append(
            SchemaDocument(
                dataset_id=dataset_id,
                kind=CandidateKind.DATASET,
                table=None,
                column=None,
                physical_name=dataset_id,
                human_name=dataset_title or dataset_id.replace("_", " "),
                description=dataset_description,
            )
        )
    for table in schema.tables.values():
        documents.append(
            SchemaDocument(
                dataset_id=dataset_id,
                kind=CandidateKind.TABLE,
                table=table.name,
                column=None,
                physical_name=table.name,
                human_name=humanize(table.name),
                aliases=table.aliases,
                description=f"Table {humanize(table.name)}",
            )
        )
        for column in table.columns.values():
            key = f"{table.name}.{column.name}"
            documents.append(
                SchemaDocument(
                    dataset_id=dataset_id,
                    kind=CandidateKind.COLUMN,
                    table=table.name,
                    column=column.name,
                    physical_name=column.name,
                    human_name=humanize(column.name),
                    aliases=column.aliases,
                    data_type=column.type,
                    description=f"{humanize(column.name)} column on {humanize(table.name)}",
                    representative_values=table_values.get(key, ()),
                )
            )
    return tuple(documents)


def humanize(identifier: str) -> str:
    return identifier.replace("_", " ")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", text.lower()))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
