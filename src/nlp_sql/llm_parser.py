"""LLM-backed semantic parsing into the safe QueryAST contract.

The model is deliberately not allowed to generate SQL. It receives the schema
and returns a small JSON query plan which is converted into QueryAST and then
validated by the normal engine pipeline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date
from typing import Any

from nlp_sql.query_ast import (
    AggregationExpression,
    ColumnExpression,
    DateRangePredicate,
    FilterExpression,
    HavingPredicate,
    JoinExpression,
    OrderBy,
    Predicate,
    QueryAST,
    QueryType,
    SortDirection,
)
from nlp_sql.conversation import ConversationContext
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.schema import SchemaRegistry


class LLMParser:
    """Parse natural language into QueryAST through an OpenAI-compatible API."""

    def __init__(
        self,
        schema: SchemaRegistry,
        *,
        base_url: str,
        api_key: str,
        model: str,
        today: date | None = None,
        timeout: float = 45.0,
        temperature: float = 0.2,
    ) -> None:
        self._schema = schema
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._today = today
        self._timeout = timeout
        self._temperature = temperature

    @classmethod
    def from_environment(
        cls, schema: SchemaRegistry, *, today: date | None = None
    ) -> LLMParser | None:
        api_key = os.getenv("NLP_SQL_LLM_API_KEY")
        if not api_key:
            return None
        return cls(
            schema,
            base_url=os.getenv("NLP_SQL_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
            model=os.getenv("NLP_SQL_LLM_MODEL", "gpt-4o-mini"),
            today=today,
            timeout=float(os.getenv("NLP_SQL_LLM_TIMEOUT", "45")),
            temperature=float(os.getenv("NLP_SQL_LLM_TEMPERATURE", "0.2")),
        )

    def parse(
        self,
        query: str,
        *,
        feedback: str | None = None,
        schema_context: tuple[dict[str, Any], ...] = (),
        conversation: ConversationContext | None = None,
    ) -> PipelineResult:
        try:
            payload = self._request(
                query,
                feedback=feedback,
                schema_context=schema_context,
                conversation=conversation,
            )
            candidate_count = 0
            selected_confidence: float | None = None
            candidates = payload.get("candidates")
            if isinstance(candidates, list) and candidates:
                valid_candidates = [item for item in candidates if isinstance(item, dict)]
                if not valid_candidates:
                    raise ValueError("LLM candidates must be objects.")
                valid_candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
                top_confidence = float(valid_candidates[0].get("confidence", 0.0))
                selected_confidence = _confidence(top_confidence)
                if len(valid_candidates) > 1:
                    second_confidence = float(valid_candidates[1].get("confidence", 0.0))
                    if top_confidence - second_confidence < 0.08:
                        return PipelineResult(
                            success=False,
                            requires_clarification=True,
                            error=QueryError(
                                "AMBIGUOUS_QUERY",
                                "The query has multiple similarly likely interpretations.",
                            ),
                            confidence=max(0.0, min(1.0, top_confidence)),
                            candidate_count=len(valid_candidates),
                        )
                selected = valid_candidates[0].get("plan")
                if not isinstance(selected, dict):
                    raise ValueError("Each LLM candidate must contain a plan object.")
                payload = selected
                candidate_count = len(valid_candidates)
            if "clarification" in payload:
                message = str(payload["clarification"]).strip()
                if message:
                    return PipelineResult(
                        success=False,
                        requires_clarification=True,
                        error=QueryError("AMBIGUOUS_QUERY", message),
                    )
            ast = self._ast_from_payload(payload)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return PipelineResult(
                success=False,
                error=QueryError("LLM_PARSE_FAILED", str(exc)),
            )
        return PipelineResult(
            success=True,
            ast=ast,
            interpretation={"source": ast.source, "status": ast.status, "parser": "llm"},
            matched_rules=("parser:llm",),
            confidence=(
                selected_confidence
                if selected_confidence is not None
                else _confidence(payload.get("confidence", 0.85))
            ),
            candidate_count=candidate_count,
        )

    def _request(
        self,
        query: str,
        *,
        feedback: str | None = None,
        schema_context: tuple[dict[str, Any], ...] = (),
        conversation: ConversationContext | None = None,
    ) -> dict[str, Any]:
        body = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(
                        feedback=feedback,
                        schema_context=schema_context,
                        conversation=conversation,
                    ),
                },
                {"role": "user", "content": query},
            ],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ValueError(f"LLM HTTP {exc.code}: {detail}") from exc
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response did not contain choices.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("LLM response did not contain message content.")
        return self._decode_json(content)

    def _system_prompt(
        self,
        *,
        feedback: str | None = None,
        schema_context: tuple[dict[str, Any], ...] = (),
        conversation: ConversationContext | None = None,
    ) -> str:
        schema = {
            table.name: {
                "aliases": list(table.aliases),
                "columns": {
                    column.name: {"type": column.type, "aliases": list(column.aliases)}
                    for column in table.columns.values()
                },
            }
            for table in self._schema.tables.values()
        }
        today = self._today.isoformat() if self._today else "unknown"
        schema_for_prompt: object = schema if not schema_context else {
            "retrieved_schema_candidates": schema_context,
            "relationships": [
                {
                    "table": foreign_key.table,
                    "column": foreign_key.column,
                    "referenced_table": foreign_key.referenced_table,
                    "referenced_column": foreign_key.referenced_column,
                }
                for table in self._schema.tables.values()
                for foreign_key in table.foreign_keys
            ],
        }
        context = json.dumps(schema_context, sort_keys=True) if schema_context else "[]"
        history = json.dumps(conversation.as_messages() if conversation else (), ensure_ascii=False)
        prompt = (
            "You are a semantic query planner. Convert the user's natural-language "
            "question into JSON matching the QueryAST contract below. Never write SQL. "
            "Use only tables and columns from the supplied schema. If the question is "
            "ambiguous or unsupported, return {\"clarification\": \"one concise question\"}. "
            f"Today is {today}. Schema context: {json.dumps(schema_for_prompt, sort_keys=True)}. "
            f"Relevant retrieved schema context: {context}. "
            f"Recent conversation context: {history}. Resolve follow-up references using it. "
            "JSON fields: type='select', source, select [{type:'column',name}], "
            "aggregations [{type:'aggregation',function,column,alias}], filters "
            "([{column,operator,value}] or {column,start,end}), having "
            "([{left:{type:'aggregation',function,column,alias},operator,value}]), "
            "where recursive filter expression {kind:'and'|'or'|'not'|'predicate', "
            "predicate:{column,operator,value}, children:[...]}, "
            "joins [{table,left_table,left_column,right_column,join_type:'INNER'|'LEFT'}], "
            "prefer declared relationships when choosing joins. "
            "group_by [column], order_by [{column,direction:'ASC'|'DESC'}], "
            "limit integer or null, status='inferred'."
            " You may instead return candidates:[{confidence,plan},...]. "
            "Return confidence between 0 and 1."
        )
        if feedback:
            prompt += (
                " The previous plan was rejected by the validator. Correct it using "
                f"this validator feedback: {feedback}"
            )
        return prompt

    def _decode_json(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        decoded = json.loads(cleaned)
        if not isinstance(decoded, dict):
            raise ValueError("LLM output must be a JSON object.")
        return decoded

    def _ast_from_payload(self, payload: dict[str, Any]) -> QueryAST:
        source = _required_string(payload, "source")
        select = tuple(
            ColumnExpression("column", _required_string(item, "name"))
            for item in _list_of_dicts(payload.get("select", []), "select")
        )
        aggregations = tuple(
            self._aggregation(item) for item in _list_of_dicts(payload.get("aggregations", []), "aggregations")
        )
        filters = tuple(self._predicate(item) for item in _list_of_dicts(payload.get("filters", []), "filters"))
        where_payload = payload.get("where")
        where = self._filter_expression(where_payload) if isinstance(where_payload, dict) else None
        having = tuple(
            HavingPredicate(
                left=self._aggregation(_required_dict(item, "left")),
                operator=_required_string(item, "operator"),
                value=item.get("value"),
            )
            for item in _list_of_dicts(payload.get("having", []), "having")
        )
        order_by = tuple(
            OrderBy(
                column=item.get("column"),
                direction=SortDirection(str(item.get("direction", "ASC")).upper()),
            )
            for item in _list_of_dicts(payload.get("order_by", []), "order_by")
        )
        joins = tuple(
            JoinExpression(
                table=_required_string(item, "table"),
                left_column=_required_string(item, "left_column"),
                right_column=_required_string(item, "right_column"),
                join_type=str(item.get("join_type", "INNER")).upper(),
                left_table=item.get("left_table"),
            )
            for item in _list_of_dicts(payload.get("joins", []), "joins")
        )
        limit = payload.get("limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
            raise ValueError("limit must be an integer or null")
        date_filters = tuple(
            DateRangePredicate(
                column=_required_string(item, "column"),
                start=date.fromisoformat(_required_string(item, "start")),
                end=date.fromisoformat(_required_string(item, "end")),
            )
            for item in _list_of_dicts(payload.get("date_filters", []), "date_filters")
        )
        return QueryAST(
            type=QueryType.SELECT,
            source=source,
            joins=joins,
            select=select or (ColumnExpression("column", "*"),),
            aggregations=aggregations,
            filters=filters + date_filters,
            where=where,
            having=having,
            group_by=tuple(str(value) for value in payload.get("group_by", [])),
            order_by=order_by,
            limit=limit,
            status=str(payload.get("status", "inferred")),
            matched_rules=("parser:llm",),
        )

    def _aggregation(self, item: dict[str, Any]) -> AggregationExpression:
        return AggregationExpression(
            type="aggregation",
            function=_required_string(item, "function").upper(),
            column=_required_string(item, "column"),
            alias=item.get("alias"),
        )

    def _predicate(self, item: dict[str, Any]) -> Predicate:
        return Predicate(
            column=_required_string(item, "column"),
            operator=_required_string(item, "operator").upper(),
            value=item.get("value"),
        )

    def _filter_expression(self, item: dict[str, Any]) -> FilterExpression:
        kind = _required_string(item, "kind").lower()
        if kind == "predicate":
            return FilterExpression(
                kind,
                predicate=self._predicate(_required_dict(item, "predicate")),
            )
        children = tuple(
            self._filter_expression(child)
            for child in _list_of_dicts(item.get("children", []), "children")
        )
        return FilterExpression(kind, children=children)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _list_of_dicts(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be an array of objects")
    return value


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
