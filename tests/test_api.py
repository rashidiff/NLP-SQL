from datetime import date

from nlp_sql.api import handle_api_request
from nlp_sql.engine import NlpSqlEngine


def test_parse_endpoint_returns_sql() -> None:
    engine = NlpSqlEngine(today=date(2026, 9, 18))

    status, body = handle_api_request(
        "/parse",
        {"query": "Show customers who spent more than 5 million"},
        engine,
    )

    assert status == 200
    assert body["success"] is True
    assert body["sql"].endswith("HAVING SUM(amount) > ?")
    assert body["parameters"] == [5_000_000]


def test_explain_endpoint_omits_sql() -> None:
    engine = NlpSqlEngine(today=date(2026, 9, 18))

    status, body = handle_api_request(
        "/explain",
        {"query": "Show the top 5 customers by total spending this month"},
        engine,
    )

    assert status == 200
    assert body["success"] is True
    assert "sql" not in body
    assert body["interpretation"]["source"] == "orders"


def test_api_rejects_missing_query() -> None:
    status, body = handle_api_request("/parse", {}, NlpSqlEngine())

    assert status == 400
    assert body["success"] is False
