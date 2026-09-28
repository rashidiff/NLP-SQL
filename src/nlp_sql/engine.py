"""Public orchestration API for parsing, validation, compilation, and explain mode."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from nlp_sql.adaptive_interpretation import AdaptiveInterpreter
from nlp_sql.compiler import SqlCompiler
from nlp_sql.datasets import (
    HOUSING_COLUMN_ALIASES,
    HOUSING_TABLE_ALIASES,
    build_housing_sqlite,
)
from nlp_sql.embedding_parser import EmbeddingParser
from nlp_sql.executor import SQLiteExecutor
from nlp_sql.parser import RuleBasedParser
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.retrieval import HybridSchemaRetriever, documents_from_schema
from nlp_sql.retrieval.base import SemanticSchemaRetriever
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry
from nlp_sql.validator import ASTValidator


class NlpSqlEngine:
    def __init__(
        self,
        schema: SchemaRegistry | None = None,
        today: date | None = None,
        executor: SQLiteExecutor | None = None,
        schema_retriever: SemanticSchemaRetriever | None = None,
    ) -> None:
        self._schema = schema or SchemaRegistry.default()
        self._parser = RuleBasedParser(self._schema, today=today)
        self._embedding_parser = EmbeddingParser(self._schema, today=today)
        self._validator = ASTValidator(self._schema)
        self._compiler = SqlCompiler()
        self._safety = SafetyValidator()
        self._executor = executor
        self._adaptive_interpreter = AdaptiveInterpreter()
        self._schema_retriever = schema_retriever or HybridSchemaRetriever(
            documents_from_schema(self._schema)
        )

    @classmethod
    def for_housing_dataset(
        cls, db_path: Path = Path("data/housing.sqlite"), today: date | None = None
    ) -> NlpSqlEngine:
        if not db_path.exists():
            build_housing_sqlite(db_path)
        schema = SchemaRegistry.from_sqlite(
            db_path,
            table_aliases=HOUSING_TABLE_ALIASES,
            column_aliases=HOUSING_COLUMN_ALIASES,
        )
        retriever = HybridSchemaRetriever.with_sqlite_store(
            documents_from_schema(
                schema,
                dataset_id="housing",
                dataset_title="Nashville Housing",
                dataset_description="Property sales and assessment records for Nashville housing.",
            ),
            db_path.with_suffix(".embeddings.sqlite"),
        )
        return cls(
            schema=schema,
            today=today,
            executor=SQLiteExecutor(db_path),
            schema_retriever=retriever,
        )

    def parse(self, query: str) -> PipelineResult:
        schema_candidates = self.search_schema(query)
        parsed = self._parse_with_hybrid_fallback(query)
        if not parsed.success or parsed.ast is None:
            return self._attach_schema_candidates(parsed, schema_candidates)
        validation_error = self._validator.validate(parsed.ast)
        if validation_error is not None:
            return PipelineResult(
                success=False,
                ast=parsed.ast,
                error=validation_error,
                schema_candidates=schema_candidates,
            )
        sql = self._compiler.compile(parsed.ast)
        safety_error = self._safety.validate_sql(sql)
        if safety_error is not None:
            return PipelineResult(
                success=False,
                ast=parsed.ast,
                error=safety_error,
                schema_candidates=schema_candidates,
            )
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            sql=sql,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
            schema_candidates=schema_candidates,
            adaptive_decision=self._adaptive_interpreter.decide(
                PipelineResult(
                    success=True,
                    ast=parsed.ast,
                    sql=sql,
                    interpretation=parsed.interpretation,
                    matched_rules=parsed.matched_rules,
                    schema_candidates=schema_candidates,
                )
            ).to_dict(),
        )

    def explain(self, query: str) -> PipelineResult:
        schema_candidates = self.search_schema(query)
        parsed = self._parse_with_hybrid_fallback(query)
        if not parsed.success or parsed.ast is None:
            return self._attach_schema_candidates(parsed, schema_candidates)
        validation_error = self._validator.validate(parsed.ast)
        if validation_error is not None:
            return PipelineResult(
                success=False,
                ast=parsed.ast,
                error=validation_error,
                schema_candidates=schema_candidates,
            )
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
            schema_candidates=schema_candidates,
            adaptive_decision=self._adaptive_interpreter.decide(
                PipelineResult(
                    success=True,
                    ast=parsed.ast,
                    interpretation=parsed.interpretation,
                    matched_rules=parsed.matched_rules,
                    schema_candidates=schema_candidates,
                )
            ).to_dict(),
        )

    def execute(self, query: str, max_rows: int = 100) -> PipelineResult:
        parsed = self.parse(query)
        if not parsed.success or parsed.sql is None:
            return parsed
        if self._executor is None:
            return PipelineResult(
                success=False,
                ast=parsed.ast,
                sql=parsed.sql,
                error=QueryError(
                    "NO_EXECUTOR",
                    "This engine was not configured with a database executor.",
                ),
                interpretation=parsed.interpretation,
                matched_rules=parsed.matched_rules,
                schema_candidates=parsed.schema_candidates,
            )
        rows = self._executor.execute(parsed.sql, max_rows=max_rows)
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            sql=parsed.sql,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
            rows=rows,
            schema_candidates=parsed.schema_candidates,
        )

    def search_schema(self, query: str, top_k: int = 8) -> tuple[dict[str, object], ...]:
        """Return schema retrieval candidates without compiling or executing SQL."""

        return tuple(
            candidate.to_dict()
            for candidate in self._schema_retriever.search(query, top_k=top_k)
        )

    def explain_schema_match(self, query: str, top_k: int = 8) -> dict[str, object]:
        return {
            "query": query,
            "candidates": list(self.search_schema(query, top_k=top_k)),
            "retrieval_stats": self.retrieval_stats(),
        }

    def retrieval_stats(self) -> dict[str, object]:
        stats = getattr(self._schema_retriever, "stats", None)
        if stats is None:
            return {}
        return dict(stats())

    def _parse_with_hybrid_fallback(self, query: str) -> PipelineResult:
        embedded = self._embedding_parser.parse(query)
        if embedded.success and embedded.ast is not None:
            validation_error = self._validator.validate(embedded.ast)
            if validation_error is None:
                return embedded
        if embedded.error is not None and embedded.error.code == "FORBIDDEN_OPERATION":
            return embedded
        return self._parser.parse(query)

    def _attach_schema_candidates(
        self,
        result: PipelineResult,
        schema_candidates: tuple[dict[str, object], ...],
    ) -> PipelineResult:
        return PipelineResult(
            success=result.success,
            ast=result.ast,
            sql=result.sql,
            error=result.error,
            requires_clarification=result.requires_clarification,
            interpretation=result.interpretation,
            matched_rules=result.matched_rules,
            rows=result.rows,
            schema_candidates=schema_candidates,
            adaptive_decision=result.adaptive_decision,
        )
