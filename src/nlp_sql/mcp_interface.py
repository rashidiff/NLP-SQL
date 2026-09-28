"""Optional MCP-style facade around the deterministic engine.

This module intentionally does not implement transport. It exposes a small
tool-like surface that an MCP server can wrap without bypassing QueryAST
validation or SQL safety validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlp_sql.dataset_registry import DatasetRegistry
from nlp_sql.engine import NlpSqlEngine


@dataclass
class NlpSqlMcpFacade:
    engine: NlpSqlEngine
    datasets: DatasetRegistry | None = None

    def list_datasets(self) -> dict[str, Any]:
        if self.datasets is None:
            return {"datasets": []}
        return {
            "datasets": [
                {
                    "id": dataset.id,
                    "title": dataset.title,
                    "description": dataset.description,
                    "source": dataset.source,
                }
                for dataset in self.datasets.list_datasets()
            ]
        }

    def search_schema(self, query: str, top_k: int = 8) -> dict[str, Any]:
        return {"query": query, "candidates": list(self.engine.search_schema(query, top_k=top_k))}

    def explain_schema_match(self, query: str, top_k: int = 8) -> dict[str, Any]:
        return self.engine.explain_schema_match(query, top_k=top_k)

    def parse_natural_language(self, query: str) -> dict[str, Any]:
        return self.engine.parse(query).to_dict()

    def compile_query(self, query: str) -> dict[str, Any]:
        return self.engine.parse(query).to_dict()

    def validate_query(self, query: str) -> dict[str, Any]:
        result = self.engine.parse(query)
        return {
            "success": result.success,
            "error": result.error.to_dict() if result.error is not None else None,
            "ast": result.ast.to_dict() if result.ast is not None else None,
        }

    def execute_query(self, query: str, max_rows: int = 100) -> dict[str, Any]:
        return self.engine.execute(query, max_rows=max_rows).to_dict()

    def explain_query(self, query: str) -> dict[str, Any]:
        return self.engine.explain(query).to_dict()

    def retrieval_stats(self) -> dict[str, Any]:
        return self.engine.retrieval_stats()
