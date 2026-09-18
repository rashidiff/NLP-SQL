"""Public orchestration API for parsing, validation, compilation, and explain mode."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from nlp_sql.compiler import SqlCompiler
from nlp_sql.datasets import (
    HOUSING_COLUMN_ALIASES,
    HOUSING_TABLE_ALIASES,
    build_housing_sqlite,
)
from nlp_sql.executor import SQLiteExecutor
from nlp_sql.parser import RuleBasedParser
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry
from nlp_sql.validator import ASTValidator


class NlpSqlEngine:
    def __init__(
        self,
        schema: SchemaRegistry | None = None,
        today: date | None = None,
        executor: SQLiteExecutor | None = None,
    ) -> None:
        self._schema = schema or SchemaRegistry.default()
        self._parser = RuleBasedParser(self._schema, today=today)
        self._validator = ASTValidator(self._schema)
        self._compiler = SqlCompiler()
        self._safety = SafetyValidator()
        self._executor = executor

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
        return cls(schema=schema, today=today, executor=SQLiteExecutor(db_path))

    def parse(self, query: str) -> PipelineResult:
        parsed = self._parser.parse(query)
        if not parsed.success or parsed.ast is None:
            return parsed
        validation_error = self._validator.validate(parsed.ast)
        if validation_error is not None:
            return PipelineResult(success=False, ast=parsed.ast, error=validation_error)
        sql = self._compiler.compile(parsed.ast)
        safety_error = self._safety.validate_sql(sql)
        if safety_error is not None:
            return PipelineResult(success=False, ast=parsed.ast, error=safety_error)
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            sql=sql,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
        )

    def explain(self, query: str) -> PipelineResult:
        parsed = self._parser.parse(query)
        if not parsed.success or parsed.ast is None:
            return parsed
        validation_error = self._validator.validate(parsed.ast)
        if validation_error is not None:
            return PipelineResult(success=False, ast=parsed.ast, error=validation_error)
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
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
            )
        rows = self._executor.execute(parsed.sql, max_rows=max_rows)
        return PipelineResult(
            success=True,
            ast=parsed.ast,
            sql=parsed.sql,
            interpretation=parsed.interpretation,
            matched_rules=parsed.matched_rules,
            rows=rows,
        )
