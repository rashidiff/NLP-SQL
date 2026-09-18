from datetime import date

from nlp_sql.date_parser import DateExpressionParser


def test_date_parser_resolves_relative_phrases() -> None:
    parser = DateExpressionParser(today=date(2026, 9, 18))

    ranges = parser.extract_ranges("show orders from last month")

    assert ranges[0].start == date(2026, 8, 1)
    assert ranges[0].end == date(2026, 9, 1)


def test_date_parser_resolves_last_n_days() -> None:
    parser = DateExpressionParser(today=date(2026, 9, 18))

    ranges = parser.extract_ranges("show orders from the last 7 days")

    assert ranges[0].start == date(2026, 9, 12)
    assert ranges[0].end == date(2026, 9, 19)


def test_date_parser_resolves_explicit_between_dates() -> None:
    parser = DateExpressionParser(today=date(2026, 9, 18))

    ranges = parser.extract_ranges("show orders between january 1 2026 and january 31 2026")

    assert ranges[0].start == date(2026, 1, 1)
    assert ranges[0].end == date(2026, 2, 1)
