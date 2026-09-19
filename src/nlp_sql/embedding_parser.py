"""TF-IDF backed parser for flexible schema matching without external AI services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from nlp_sql.date_parser import DateExpressionParser
from nlp_sql.normalizer import TextNormalizer
from nlp_sql.query_ast import (
    AggregationExpression,
    ColumnExpression,
    OrderBy,
    Predicate,
    QueryAST,
    QueryType,
    SortDirection,
)
from nlp_sql.result import PipelineResult, QueryError
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry

TABLE_SCORE_THRESHOLD = 0.35
COLUMN_SCORE_THRESHOLD = 0.35


@dataclass(frozen=True)
class _SchemaDocument:
    kind: str
    table: str
    column: str | None
    text: str


@dataclass(frozen=True)
class _ScoredDocument:
    document: _SchemaDocument
    score: float


@dataclass(frozen=True)
class _OperatorValue:
    operator: str
    value: int | float
    start: int


class EmbeddingParser:
    """Parse flexible user phrasing by ranking schema candidates with local TF-IDF."""

    _AGGREGATIONS: tuple[tuple[str, str], ...] = (
        ("average", "AVG"),
        ("avg", "AVG"),
        ("total", "SUM"),
        ("sum", "SUM"),
        ("count", "COUNT"),
        ("number", "COUNT"),
        ("maximum", "MAX"),
        ("highest", "MAX"),
        ("minimum", "MIN"),
        ("lowest", "MIN"),
    )
    _HIGH_CUES = {"expensive", "pricey", "costly", "many", "large", "big", "highest", "top"}
    _LOW_CUES = {"cheap", "cheapest", "affordable", "small", "lowest", "bottom"}
    _RECENT_CUES = {"recent", "latest", "newest"}
    _OLD_CUES = {"oldest", "old"}
    _LAND_CUES = {"land", "acre", "acres", "acreage", "lot"}
    _ROOM_CUES = {"room", "rooms", "bedroom", "bedrooms", "beds"}
    _PRICE_CUES = {"expensive", "cheap", "pricey", "costly", "affordable"}

    def __init__(
        self,
        schema: SchemaRegistry,
        today: date | None = None,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self._schema = schema
        self._normalizer = normalizer or TextNormalizer()
        self._date_parser = DateExpressionParser(today=today)
        self._safety = SafetyValidator()
        self._documents = self._build_documents(schema)
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._matrix = self._vectorizer.fit_transform(document.text for document in self._documents)

    def parse(self, query: str) -> PipelineResult:
        normalized = self._normalizer.normalize(query)
        safety_error = self._safety.validate_input(normalized.text)
        if safety_error is not None:
            return PipelineResult(success=False, error=safety_error)

        tokens = normalized.text.split()
        phrases = self._phrases(tokens, max_words=4)
        source_match = self._best_match(phrases, kind="table")
        if source_match is None or source_match.score < TABLE_SCORE_THRESHOLD:
            return self._low_confidence("Could not resolve a table with enough confidence.")

        source = source_match.document.table
        if not self._has_flexible_semantic_cue(tokens, source):
            return self._low_confidence("No flexible semantic cue was detected.")

        numeric_events = self._operator_values(tokens)
        aggregation = self._detect_aggregation(tokens, source, phrases)
        filters = self._build_filters(source, phrases, numeric_events)
        order_by = self._build_order_by(tokens, source, phrases, aggregation, numeric_events)
        limit = self._detect_limit(tokens)

        for date_range in self._date_parser.extract_ranges(normalized.text):
            column = self._date_column(source)
            filters.append(
                Predicate(column, ">=", date_range.start.isoformat())
                if date_range.start != date.min
                else Predicate(column, "<", date_range.end.isoformat())
            )

        if aggregation is None and not filters and not order_by and limit is None:
            return self._low_confidence("No flexible query operation was inferred.")

        select = () if aggregation is not None else (ColumnExpression("column", "*"),)
        ast = QueryAST(
            type=QueryType.SELECT,
            source=source,
            select=select,
            aggregations=(aggregation,) if aggregation is not None else (),
            filters=tuple(filters),
            order_by=tuple(order_by),
            limit=limit,
            status="inferred",
            matched_rules=self._matched_rules(source_match, filters, order_by, aggregation),
        )
        return PipelineResult(
            success=True,
            ast=ast,
            interpretation=self._interpret(ast),
            matched_rules=ast.matched_rules,
        )

    def _build_filters(
        self, source: str, phrases: tuple[str, ...], numeric_events: tuple[_OperatorValue, ...]
    ) -> list[Predicate]:
        filters: list[Predicate] = []
        used_columns: set[str] = set()
        for event in numeric_events:
            column_match = self._best_column_near(source, phrases, event.start)
            if column_match is None or column_match.score < COLUMN_SCORE_THRESHOLD:
                column_match = self._best_numeric_column(source, phrases)
            if column_match is None or column_match.score < COLUMN_SCORE_THRESHOLD:
                continue
            column = column_match.document.column
            if column is None or column in used_columns:
                continue
            filters.append(Predicate(column, event.operator, event.value))
            used_columns.add(column)
        return filters

    def _has_flexible_semantic_cue(self, tokens: list[str], source: str) -> bool:
        token_set = set(tokens)
        if token_set & (self._PRICE_CUES | {"pricey", "costly", "affordable"}):
            return True
        if token_set & {"many"} and token_set & self._ROOM_CUES:
            return True
        if token_set & {"large", "big"} and token_set & self._LAND_CUES:
            return True
        return bool(token_set & (self._RECENT_CUES | self._OLD_CUES)) and self._schema.has_column(
            source, self._date_column(source)
        )

    def _build_order_by(
        self,
        tokens: list[str],
        source: str,
        phrases: tuple[str, ...],
        aggregation: AggregationExpression | None,
        numeric_events: tuple[_OperatorValue, ...],
    ) -> list[OrderBy]:
        token_set = set(tokens)
        if aggregation is not None or numeric_events:
            return []
        direction = SortDirection.ASC if token_set & self._LOW_CUES else SortDirection.DESC
        if token_set & self._RECENT_CUES:
            column = self._date_column(source)
            return [OrderBy(column=column, direction=SortDirection.DESC)]
        if token_set & self._OLD_CUES:
            column = self._date_column(source)
            return [OrderBy(column=column, direction=SortDirection.ASC)]

        column_match = self._semantic_column(source, token_set, phrases)
        if column_match is None or column_match.score < COLUMN_SCORE_THRESHOLD:
            return []
        matched_column = column_match.document.column
        if matched_column is None:
            return []
        return [OrderBy(column=matched_column, direction=direction)]

    def _semantic_column(
        self, source: str, token_set: set[str], phrases: tuple[str, ...]
    ) -> _ScoredDocument | None:
        if token_set & self._PRICE_CUES:
            return self._column_by_name(source, ("sale_price", "amount", "total_value", "price"))
        if token_set & self._ROOM_CUES:
            return self._column_by_name(source, ("bedrooms", "full_bath"))
        if token_set & self._LAND_CUES:
            return self._column_by_name(source, ("acreage", "land_value", "total_value"))
        return self._best_numeric_column(source, phrases)

    def _detect_aggregation(
        self, tokens: list[str], source: str, phrases: tuple[str, ...]
    ) -> AggregationExpression | None:
        function = None
        token_set = set(tokens)
        for cue, candidate in self._AGGREGATIONS:
            if cue in token_set:
                function = candidate
                break
        if function is None:
            return None
        if function == "COUNT":
            column = self._count_column(source)
        else:
            column_match = self._best_numeric_column(source, phrases)
            if column_match is None or column_match.score < COLUMN_SCORE_THRESHOLD:
                return None
            matched_column = column_match.document.column
            if matched_column is None:
                return None
            column = matched_column
        if column is None:
            return None
        alias = "count" if function == "COUNT" else f"{function.lower()}_{column}"
        return AggregationExpression("aggregation", function, column, alias)

    def _operator_values(self, tokens: list[str]) -> tuple[_OperatorValue, ...]:
        events: list[_OperatorValue] = []
        for index, token in enumerate(tokens):
            value = self._number(token)
            if value is None:
                continue
            before = tokens[max(0, index - 3) : index]
            operator = self._operator_before(before)
            if operator is not None:
                events.append(_OperatorValue(operator, value, index))
        return tuple(events)

    def _operator_before(self, before: list[str]) -> str | None:
        phrase = " ".join(before)
        if "between" in before:
            return "="
        if phrase.endswith("at least") or phrase.endswith("minimum"):
            return ">="
        if phrase.endswith("at most") or phrase.endswith("maximum"):
            return "<="
        if phrase.endswith("less than") or phrase.endswith("under") or phrase.endswith("below"):
            return "<"
        if (
            phrase.endswith("greater than")
            or phrase.endswith("more than")
            or phrase.endswith("above")
        ):
            return ">"
        if phrase.endswith("after"):
            return ">"
        if phrase.endswith("before"):
            return "<"
        if phrase.endswith("equals") or phrase.endswith("equal to") or phrase.endswith("is"):
            return "="
        return None

    def _detect_limit(self, tokens: list[str]) -> int | None:
        for index, token in enumerate(tokens[:-1]):
            if token in {"top", "bottom", "first", "latest", "oldest"}:
                value = self._number(tokens[index + 1])
                if isinstance(value, int):
                    return value
        return None

    def _best_match(self, phrases: tuple[str, ...], kind: str) -> _ScoredDocument | None:
        matches = self._rank_phrases(phrases)
        for match in matches:
            if match.document.kind == kind:
                return match
        return None

    def _best_numeric_column(self, source: str, phrases: tuple[str, ...]) -> _ScoredDocument | None:
        for match in self._rank_phrases(phrases):
            if match.document.kind != "column" or match.document.table != source:
                continue
            column = match.document.column
            if column is None:
                continue
            column_meta = self._schema.get_column(source, column)
            if column_meta is not None and column_meta.type in {"integer", "decimal", "datetime"}:
                return match
        return None

    def _best_column_near(
        self, source: str, phrases: tuple[str, ...], value_position: int
    ) -> _ScoredDocument | None:
        nearby = tuple(
            phrase
            for phrase in phrases
            if self._phrase_distance(phrase, value_position) <= 4
        )
        return self._best_numeric_column(source, nearby or phrases)

    def _column_by_name(self, source: str, names: tuple[str, ...]) -> _ScoredDocument | None:
        for name in names:
            if self._schema.has_column(source, name):
                document = _SchemaDocument("column", source, name, name)
                return _ScoredDocument(document, 1.0)
        return None

    def _rank_phrases(self, phrases: tuple[str, ...]) -> tuple[_ScoredDocument, ...]:
        if not phrases:
            return ()
        query_matrix = self._vectorizer.transform(phrases)
        scores = cosine_similarity(query_matrix, self._matrix)
        best_by_document: dict[int, float] = {}
        for row in range(scores.shape[0]):
            for column in range(scores.shape[1]):
                score = float(scores[row, column])
                best_by_document[column] = max(best_by_document.get(column, 0.0), score)
        ranked = [
            _ScoredDocument(self._documents[index], score)
            for index, score in best_by_document.items()
            if score > 0
        ]
        return tuple(sorted(ranked, key=lambda item: item.score, reverse=True))

    def _matched_rules(
        self,
        source_match: _ScoredDocument,
        filters: list[Predicate],
        order_by: list[OrderBy],
        aggregation: AggregationExpression | None,
    ) -> tuple[str, ...]:
        rules = [f"embedding:table:{source_match.document.table}:{source_match.score:.2f}"]
        rules.extend(f"embedding:filter:{predicate.column}" for predicate in filters)
        rules.extend(f"embedding:order:{order.column}" for order in order_by if order.column)
        if aggregation is not None:
            rules.append(f"embedding:aggregation:{aggregation.function}:{aggregation.column}")
        return tuple(rules)

    def _interpret(self, ast: QueryAST) -> dict[str, Any]:
        interpretation: dict[str, Any] = {"source": ast.source, "status": ast.status}
        if ast.aggregations:
            interpretation["aggregation"] = ast.aggregations[0].function
            interpretation["metric"] = ast.aggregations[0].column
        elif ast.order_by:
            interpretation["metric"] = ast.order_by[0].column
            interpretation["direction"] = ast.order_by[0].direction.value
        elif ast.filters:
            predicate = ast.filters[0]
            if isinstance(predicate, Predicate):
                interpretation["metric"] = predicate.column
                interpretation["operator"] = predicate.operator
                interpretation["value"] = predicate.value
        return interpretation

    def _low_confidence(self, message: str) -> PipelineResult:
        return PipelineResult(success=False, error=QueryError("LOW_CONFIDENCE", message))

    def _date_column(self, source: str) -> str:
        for candidate in ("created_at", "sale_date", "date", "year_built"):
            if self._schema.has_column(source, candidate):
                return candidate
        return "created_at"

    def _count_column(self, source: str) -> str:
        for candidate in ("id", "unique_id"):
            if self._schema.has_column(source, candidate):
                return candidate
        return next(iter(self._schema.tables[source].columns))

    def _number(self, token: str) -> int | float | None:
        try:
            value = float(token)
        except ValueError:
            return None
        return int(value) if value.is_integer() else value

    def _phrase_distance(self, phrase: str, value_position: int) -> int:
        size = len(phrase.split())
        return min(abs(value_position - index) for index in range(max(size, value_position + 1)))

    def _phrases(self, tokens: list[str], max_words: int) -> tuple[str, ...]:
        phrases: list[str] = []
        for size in range(min(max_words, len(tokens)), 0, -1):
            for index in range(0, len(tokens) - size + 1):
                phrases.append(" ".join(tokens[index : index + size]))
        return tuple(phrases)

    def _build_documents(self, schema: SchemaRegistry) -> tuple[_SchemaDocument, ...]:
        documents: list[_SchemaDocument] = []
        for table in schema.tables.values():
            documents.append(
                _SchemaDocument("table", table.name, None, self._document_text(table.aliases))
            )
            for column in table.columns.values():
                documents.append(
                    _SchemaDocument(
                        "column",
                        table.name,
                        column.name,
                        self._document_text((column.name, *column.aliases, column.type)),
                    )
                )
        return tuple(documents)

    def _document_text(self, parts: tuple[str, ...]) -> str:
        expanded = []
        for part in parts:
            expanded.append(part)
            expanded.append(part.replace("_", " "))
        return " ".join(dict.fromkeys(expanded))
