"""Tokenization and deterministic multi-word phrase matching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PhraseKind(StrEnum):
    OPERATOR = "operator"
    DATE = "date"
    ORDERING = "ordering"
    GROUPING = "grouping"
    LIMIT = "limit"


@dataclass(frozen=True)
class Token:
    text: str
    position: int


@dataclass(frozen=True)
class PhraseRule:
    phrase: str
    kind: PhraseKind
    value: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.phrase.split())


@dataclass(frozen=True)
class PhraseMatch:
    phrase: str
    kind: PhraseKind
    value: str
    start: int
    end: int


DEFAULT_PHRASE_RULES: tuple[PhraseRule, ...] = (
    PhraseRule("not equal to", PhraseKind.OPERATOR, "!="),
    PhraseRule("greater than", PhraseKind.OPERATOR, ">"),
    PhraseRule("more than", PhraseKind.OPERATOR, ">"),
    PhraseRule("less than", PhraseKind.OPERATOR, "<"),
    PhraseRule("at least", PhraseKind.OPERATOR, ">="),
    PhraseRule("at most", PhraseKind.OPERATOR, "<="),
    PhraseRule("equal to", PhraseKind.OPERATOR, "="),
    PhraseRule("is not", PhraseKind.OPERATOR, "!="),
    PhraseRule("last month", PhraseKind.DATE, "LAST_MONTH"),
    PhraseRule("previous month", PhraseKind.DATE, "LAST_MONTH"),
    PhraseRule("last week", PhraseKind.DATE, "LAST_WEEK"),
    PhraseRule("previous week", PhraseKind.DATE, "LAST_WEEK"),
    PhraseRule("last year", PhraseKind.DATE, "LAST_YEAR"),
    PhraseRule("previous year", PhraseKind.DATE, "LAST_YEAR"),
    PhraseRule("this month", PhraseKind.DATE, "THIS_MONTH"),
    PhraseRule("this week", PhraseKind.DATE, "THIS_WEEK"),
    PhraseRule("this year", PhraseKind.DATE, "THIS_YEAR"),
    PhraseRule("order by", PhraseKind.ORDERING, "ORDER_BY"),
    PhraseRule("sort by", PhraseKind.ORDERING, "ORDER_BY"),
    PhraseRule("group by", PhraseKind.GROUPING, "GROUP_BY"),
    PhraseRule("top", PhraseKind.LIMIT, "TOP"),
    PhraseRule("bottom", PhraseKind.LIMIT, "BOTTOM"),
)


class Tokenizer:
    """Whitespace tokenizer for already-normalized English query text."""

    def tokenize(self, text: str) -> tuple[Token, ...]:
        return tuple(Token(token, index) for index, token in enumerate(text.split()))


class PhraseMatcher:
    """Match configured phrases using longest-match-first semantics."""

    def __init__(self, rules: tuple[PhraseRule, ...] = DEFAULT_PHRASE_RULES) -> None:
        self._rules = tuple(sorted(rules, key=lambda rule: len(rule.tokens), reverse=True))

    def find_matches(self, tokens: tuple[Token, ...]) -> tuple[PhraseMatch, ...]:
        matches: list[PhraseMatch] = []
        index = 0
        token_text = tuple(token.text for token in tokens)

        while index < len(tokens):
            match = self._match_at(token_text, index)
            if match is None:
                index += 1
                continue
            matches.append(match)
            index = match.end

        return tuple(matches)

    def _match_at(self, token_text: tuple[str, ...], index: int) -> PhraseMatch | None:
        for rule in self._rules:
            end = index + len(rule.tokens)
            if token_text[index:end] == rule.tokens:
                return PhraseMatch(
                    phrase=rule.phrase,
                    kind=rule.kind,
                    value=rule.value,
                    start=index,
                    end=end,
                )
        return None
