from decimal import Decimal

import pytest

from nlp_sql.number_parser import NumberParseError, normalize_number_value, parse_number_expression


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ten", Decimal(10)),
        ("twenty five", Decimal(25)),
        ("one hundred", Decimal(100)),
        ("ten thousand", Decimal(10_000)),
        ("one hundred thousand", Decimal(100_000)),
        ("one million", Decimal(1_000_000)),
        ("five million", Decimal(5_000_000)),
        ("one and a half million", Decimal(1_500_000)),
        ("2.5 million", Decimal(2_500_000)),
        ("3 billion", Decimal(3_000_000_000)),
        ("5,000,000", Decimal(5_000_000)),
        ("$5,000,000", Decimal(5_000_000)),
    ],
)
def test_parse_number_expression(text: str, expected: Decimal) -> None:
    assert parse_number_expression(text) == expected


def test_normalize_number_value_returns_int_for_whole_numbers() -> None:
    assert normalize_number_value(Decimal("5000000")) == 5_000_000


def test_normalize_number_value_returns_float_for_fractional_numbers() -> None:
    assert normalize_number_value(Decimal("12.5")) == 12.5


def test_parse_number_expression_rejects_unknown_words() -> None:
    with pytest.raises(NumberParseError):
        parse_number_expression("many millions")
