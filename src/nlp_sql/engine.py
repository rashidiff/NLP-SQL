"""Public orchestration API for parsing, validation, compilation, and explain mode."""

from __future__ import annotations

from datetime import date

from nlp_sql.compiler import SqlCompiler
from nlp_sql.parser import RuleBasedParser
from nlp_sql.result import PipelineResult
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry
from nlp_sql.validator import ASTValidator


class NlpSqlEngine:
    def __init__(self, schema: SchemaRegistry | None = None, today: date | None = None) -> None:
        self._schema = schema or SchemaRegistry.default()
        self._parser = RuleBasedParser(self._schema, today=today)
        self._validator = ASTValidator(self._schema)
        self._compiler = SqlCompiler()
        self._safety = SafetyValidator()

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
