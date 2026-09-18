"""English text normalization for business queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nlp_sql.number_parser import (
    NUMBER_WORDS,
    NumberParseError,
    normalize_number_value,
    parse_number_expression,
)


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    number_replacements: tuple[tuple[str, int | float], ...]


class TextNormalizer:
    """Normalize English query text without introducing SQL fragments."""

    _CONTRACTIONS: dict[str, str] = {
        "what's": "what is",
        "who's": "who is",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "can't": "cannot",
    }
    _PUNCTUATION_RE = re.compile(r"[;:!?()\[\]{}\"`]")
    _NUMERIC_RE = re.compile(
        r"\$?\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:thousand|million|billion))?\b",
        re.IGNORECASE,
    )

    def normalize(self, text: str) -> NormalizedText:
        original = text
        normalized = text.lower()
        normalized = self._expand_contractions(normalized)
        normalized = normalized.replace("-", " ")
        normalized = self._PUNCTUATION_RE.sub(" ", normalized)
        normalized = normalized.replace(",", "")
        normalized, numeric_replacements = self._replace_numeric_literals(normalized)
        normalized, word_replacements = self._replace_word_numbers(normalized)
        normalized = self._collapse_whitespace(normalized)
        return NormalizedText(
            original=original,
            text=normalized,
            number_replacements=tuple(numeric_replacements + word_replacements),
        )

    def _expand_contractions(self, text: str) -> str:
        for contraction, expanded in self._CONTRACTIONS.items():
            text = text.replace(contraction, expanded)
        return text

    def _replace_numeric_literals(self, text: str) -> tuple[str, list[tuple[str, int | float]]]:
        replacements: list[tuple[str, int | float]] = []

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                value = normalize_number_value(parse_number_expression(raw))
            except NumberParseError:
                return raw
            replacements.append((raw, value))
            return str(value)

        return self._NUMERIC_RE.sub(replace, text), replacements

    def _replace_word_numbers(self, text: str) -> tuple[str, list[tuple[str, int | float]]]:
        tokens = text.split()
        output: list[str] = []
        replacements: list[tuple[str, int | float]] = []
        index = 0

        while index < len(tokens):
            if tokens[index] not in NUMBER_WORDS or tokens[index] == "and":
                output.append(tokens[index])
                index += 1
                continue

            best_end: int | None = None
            best_value: int | float | None = None
            upper_bound = min(len(tokens), index + 8)
            for end in range(upper_bound, index, -1):
                phrase_tokens = tokens[index:end]
                if any(token not in NUMBER_WORDS for token in phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                try:
                    best_value = normalize_number_value(parse_number_expression(phrase))
                    best_end = end
                    break
                except NumberParseError:
                    continue

            if best_end is None or best_value is None:
                output.append(tokens[index])
                index += 1
                continue

            phrase = " ".join(tokens[index:best_end])
            output.append(str(best_value))
            replacements.append((phrase, best_value))
            index = best_end

        return " ".join(output), replacements

    def _collapse_whitespace(self, text: str) -> str:
        return " ".join(text.split())
