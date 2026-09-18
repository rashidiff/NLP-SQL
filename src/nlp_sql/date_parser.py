"""Rule-based English date expression parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    matched_text: str
    rule: str


class DateExpressionParser:
    """Resolve supported English date expressions into explicit ranges."""

    _MONTH_RE = re.compile(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(\d{1,2})(?:\s+(\d{4}))?\b"
    )
    _RELATIVE_DAYS_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+days\b")

    def __init__(self, today: date | None = None) -> None:
        self._today = today or date.today()

    def extract_ranges(self, text: str) -> tuple[DateRange, ...]:
        ranges: list[DateRange] = []
        for phrase, builder in (
            ("today", self._today_range),
            ("yesterday", self._yesterday_range),
            ("this week", self._this_week),
            ("last week", self._last_week),
            ("previous week", self._last_week),
            ("this month", self._this_month),
            ("last month", self._last_month),
            ("previous month", self._last_month),
            ("this year", self._this_year),
            ("last year", self._last_year),
            ("previous year", self._last_year),
        ):
            if phrase in text:
                start, end = builder()
                ranges.append(DateRange(start, end, phrase, phrase.replace(" ", "_").upper()))

        for match in self._RELATIVE_DAYS_RE.finditer(text):
            days = int(match.group(1))
            ranges.append(
                DateRange(
                    self._today - timedelta(days=days - 1),
                    self._today + timedelta(days=1),
                    match.group(0),
                    "RELATIVE_DAYS",
                )
            )

        between = self._extract_between_dates(text)
        if between is not None:
            ranges.append(between)

        for keyword, rule in (("after", "AFTER_DATE"), ("before", "BEFORE_DATE")):
            parsed = self._extract_single_bound(text, keyword, rule)
            if parsed is not None:
                ranges.append(parsed)

        return tuple(ranges)

    def _today_range(self) -> tuple[date, date]:
        return self._today, self._today + timedelta(days=1)

    def _yesterday_range(self) -> tuple[date, date]:
        start = self._today - timedelta(days=1)
        return start, self._today

    def _this_week(self) -> tuple[date, date]:
        start = self._today - timedelta(days=self._today.weekday())
        return start, start + timedelta(days=7)

    def _last_week(self) -> tuple[date, date]:
        this_start, _ = self._this_week()
        start = this_start - timedelta(days=7)
        return start, this_start

    def _this_month(self) -> tuple[date, date]:
        start = self._today.replace(day=1)
        return start, _add_month(start)

    def _last_month(self) -> tuple[date, date]:
        this_start = self._today.replace(day=1)
        previous_start = _add_month(this_start, -1)
        return previous_start, this_start

    def _this_year(self) -> tuple[date, date]:
        start = self._today.replace(month=1, day=1)
        return start, start.replace(year=start.year + 1)

    def _last_year(self) -> tuple[date, date]:
        start = self._today.replace(year=self._today.year - 1, month=1, day=1)
        return start, start.replace(year=start.year + 1)

    def _extract_between_dates(self, text: str) -> DateRange | None:
        pattern = re.compile(
            r"\b(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)(?:$|\s+(?:order|sort|by|top|bottom))"
        )
        match = pattern.search(text)
        if not match:
            return None
        start = self._parse_date(match.group(1))
        end = self._parse_date(match.group(2))
        if start is None or end is None:
            return None
        return DateRange(start, end + timedelta(days=1), match.group(0), "BETWEEN_DATES")

    def _extract_single_bound(self, text: str, keyword: str, rule: str) -> DateRange | None:
        match = re.search(rf"\b{keyword}\s+(.+)$", text)
        if not match:
            return None
        parsed = self._parse_date(match.group(1))
        if parsed is None:
            return None
        if keyword == "after":
            return DateRange(parsed + timedelta(days=1), date.max, match.group(0), rule)
        return DateRange(date.min, parsed, match.group(0), rule)

    def _parse_date(self, text: str) -> date | None:
        match = self._MONTH_RE.search(text)
        if not match:
            return None
        month_name, day_text, year_text = match.groups()
        month = _MONTHS[month_name]
        year = int(year_text) if year_text else self._today.year
        return date(year, month, int(day_text))


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _add_month(value: date, months: int = 1) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month)
