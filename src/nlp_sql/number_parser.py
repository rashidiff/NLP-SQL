"""Deterministic English number parsing."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


class NumberParseError(ValueError):
    """Raised when a number expression cannot be parsed safely."""


_UNITS: dict[str, Decimal] = {
    "zero": Decimal(0),
    "one": Decimal(1),
    "two": Decimal(2),
    "three": Decimal(3),
    "four": Decimal(4),
    "five": Decimal(5),
    "six": Decimal(6),
    "seven": Decimal(7),
    "eight": Decimal(8),
    "nine": Decimal(9),
    "ten": Decimal(10),
    "eleven": Decimal(11),
    "twelve": Decimal(12),
    "thirteen": Decimal(13),
    "fourteen": Decimal(14),
    "fifteen": Decimal(15),
    "sixteen": Decimal(16),
    "seventeen": Decimal(17),
    "eighteen": Decimal(18),
    "nineteen": Decimal(19),
}

_TENS: dict[str, Decimal] = {
    "twenty": Decimal(20),
    "thirty": Decimal(30),
    "forty": Decimal(40),
    "fifty": Decimal(50),
    "sixty": Decimal(60),
    "seventy": Decimal(70),
    "eighty": Decimal(80),
    "ninety": Decimal(90),
}

_SCALES: dict[str, Decimal] = {
    "hundred": Decimal(100),
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
}

_ALLOWED_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and", "a", "half"}
_NUMERIC_RE = re.compile(
    r"^\$?(?P<number>\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>thousand|million|billion))?$",
    re.IGNORECASE,
)


def parse_number_expression(text: str) -> Decimal:
    """Parse a supported numeric or English number expression.

    Supported examples include ``5,000,000``, ``$5,000,000``, ``2.5 million``,
    ``twenty five``, ``one hundred thousand``, and ``one and a half million``.
    """

    expression = _clean_number_text(text)
    if not expression:
        raise NumberParseError("Number expression is empty.")

    numeric = _parse_numeric_expression(expression)
    if numeric is not None:
        return numeric

    tokens = expression.replace("-", " ").split()
    unknown = [token for token in tokens if token not in _ALLOWED_WORDS]
    if unknown:
        raise NumberParseError(f"Unsupported number token: {unknown[0]}")

    return _parse_word_tokens(tokens)


def normalize_number_value(value: Decimal) -> int | float:
    """Return an int for whole numbers, otherwise a float for JSON friendliness."""

    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _clean_number_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("$", "$").split())


def _parse_numeric_expression(expression: str) -> Decimal | None:
    match = _NUMERIC_RE.fullmatch(expression)
    if not match:
        return None
    number_text = match.group("number").replace(",", "")
    try:
        value = Decimal(number_text)
    except InvalidOperation as exc:
        raise NumberParseError(f"Invalid numeric expression: {expression}") from exc
    scale = match.group("scale")
    if scale:
        value *= _SCALES[scale.lower()]
    return value


def _parse_word_tokens(tokens: list[str]) -> Decimal:
    if tokens == ["a", "half"] or tokens == ["half"]:
        return Decimal("0.5")

    if "half" in tokens:
        return _parse_fractional_scale(tokens)

    total = Decimal(0)
    current = Decimal(0)
    saw_number = False

    for token in tokens:
        if token == "and":
            continue
        if token in _UNITS:
            current += _UNITS[token]
            saw_number = True
        elif token in _TENS:
            current += _TENS[token]
            saw_number = True
        elif token == "hundred":
            current = (current or Decimal(1)) * _SCALES[token]
            saw_number = True
        elif token in {"thousand", "million", "billion"}:
            total += (current or Decimal(1)) * _SCALES[token]
            current = Decimal(0)
            saw_number = True
        elif token == "a":
            continue
        else:
            raise NumberParseError(f"Unsupported number token: {token}")

    if not saw_number:
        raise NumberParseError("No number found.")
    return total + current


def _parse_fractional_scale(tokens: list[str]) -> Decimal:
    if tokens[-2:] == ["a", "half"]:
        raise NumberParseError("Fractional scale is missing.")
    if tokens[-1] not in {"thousand", "million", "billion"}:
        raise NumberParseError("Half expressions require a scale.")

    scale = _SCALES[tokens[-1]]
    prefix = tokens[:-1]
    if prefix[-3:] == ["and", "a", "half"]:
        whole_tokens = prefix[:-3]
        whole = _parse_word_tokens(whole_tokens) if whole_tokens else Decimal(0)
        return (whole + Decimal("0.5")) * scale
    if prefix[-2:] == ["a", "half"]:
        whole_tokens = prefix[:-2]
        whole = _parse_word_tokens(whole_tokens) if whole_tokens else Decimal(0)
        return (whole + Decimal("0.5")) * scale
    raise NumberParseError("Unsupported half expression.")
