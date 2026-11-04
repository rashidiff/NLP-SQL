"""Standard MCP transport for the safe NLP-SQL engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nlp_sql.engine import NlpSqlEngine


def build_server(engine: NlpSqlEngine) -> Any:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("nlp-sql")

    @server.tool()
    def inspect_schema(query: str = "") -> dict[str, Any]:
        """Inspect schema candidates relevant to a natural-language question."""

        return engine.explain_schema_match(query)

    @server.tool()
    def search_schema(query: str, top_k: int = 8) -> dict[str, Any]:
        """Search tables, columns, aliases and representative values."""

        return {"query": query, "candidates": list(engine.search_schema(query, top_k=top_k))}

    @server.tool()
    def plan_query(query: str) -> dict[str, Any]:
        """Create and rank local non-LLM QueryAST candidates."""

        return engine.explain(query).to_dict()

    @server.tool()
    def validate_plan(query: str) -> dict[str, Any]:
        """Validate the local plan without executing it."""

        result = engine.parse(query)
        return {
            "success": result.success,
            "ast": result.ast.to_dict() if result.ast is not None else None,
            "error": result.error.to_dict() if result.error is not None else None,
        }

    @server.tool()
    def explain_query(query: str) -> dict[str, Any]:
        """Explain semantic interpretation, validation and compiled SQL."""

        return engine.parse(query).to_dict()

    @server.tool()
    def execute_query(query: str, max_rows: int = 100) -> dict[str, Any]:
        """Execute a validated read-only query with a bounded result set."""

        return engine.execute(query, max_rows=max_rows).to_dict()

    @server.tool()
    def record_feedback(query: str, accepted: bool, score: float = 1.0) -> dict[str, Any]:
        """Store explicit user feedback for future local semantic ranking."""

        result = engine.parse(query)
        engine.record_feedback(
            query,
            success=accepted,
            plan=result.ast,
            score=score,
            error_code=result.error.code if result.error is not None else None,
        )
        return {"recorded": True, "query": query, "accepted": accepted}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLP-SQL as an MCP server over stdio.")
    parser.add_argument("--db", default="data/housing.sqlite")
    args = parser.parse_args()
    engine = NlpSqlEngine.for_housing_dataset(db_path=Path(args.db))
    build_server(engine).run(transport="stdio")
