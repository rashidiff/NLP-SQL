from datetime import date

import pytest

from nlp_sql.engine import NlpSqlEngine


@pytest.fixture
def engine() -> NlpSqlEngine:
    return NlpSqlEngine(today=date(2026, 9, 18))


@pytest.mark.parametrize(
    "query",
    [
        "Show all customers",
        "Show all orders",
        "Show the first 10 customers",
        "Show orders above 5 million",
        "Show orders greater than 1000",
        "Show orders less than 500",
        "Show orders between 100 and 500",
        "How many customers are there?",
        "How many orders were placed today?",
        "How many orders were placed last month?",
        "What is the total revenue this month?",
        "What was the total revenue last month?",
        "What is the average order amount?",
        "Show customers who spent more than 10000",
        "Show customers who spent at least 5000",
        "Show the 5 customers with the highest total spending",
        "Show the 10 customers with the lowest total spending",
        "Show the latest 20 orders",
        "Show the oldest 10 orders",
        "Sort orders by amount descending",
        "Sort customers by name ascending",
        "Count orders per customer",
        "Show total sales per customer",
        "Show average spending per customer",
        "Show orders from yesterday",
        "Show orders from the last 7 days",
        "Show orders from the last 30 days",
        "Show orders after January 1 2026",
        "Show orders before March 1 2026",
        "Show orders between January 1 2026 and January 31 2026",
        "Show the maximum order value",
        "Show the minimum order value this month",
        "Total sales this month",
        "Average order amount",
        "Maximum order amount this year",
        "Minimum purchase amount",
        "Number of orders per customer",
        "Total revenue by customer",
        "Sales by month",
        "Show the top 5 customers by total spending this month",
    ],
)
def test_supported_queries_parse_to_parameterized_sql(engine: NlpSqlEngine, query: str) -> None:
    result = engine.parse(query)

    assert result.success, query
    assert result.ast is not None
    assert result.sql is not None
    assert "?" in result.sql.sql or not result.sql.parameters
    assert ";" not in result.sql.sql


def test_core_example_generates_grouped_having_sql(engine: NlpSqlEngine) -> None:
    result = engine.parse("Show customers who spent more than 5 million")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT customer_id, SUM(amount) AS total_purchase FROM orders "
        "GROUP BY customer_id HAVING SUM(amount) > ?"
    )
    assert result.sql.parameters == (5_000_000,)


def test_explain_mode_returns_interpretation_without_sql(engine: NlpSqlEngine) -> None:
    result = engine.explain("Show the top 5 customers by total spending this month")

    assert result.success
    assert result.sql is None
    assert result.interpretation is not None
    assert result.interpretation["source"] == "orders"
    assert result.matched_rules


@pytest.mark.parametrize(
    "query",
    [
        "drop customers",
        "delete all customers",
        "DROP TABLE users",
        "show users; DROP TABLE users",
        "run arbitrary SQL",
    ],
)
def test_security_or_unsupported_queries_are_rejected(engine: NlpSqlEngine, query: str) -> None:
    result = engine.parse(query)

    assert not result.success


def test_ambiguous_high_value_customers_requires_clarification(engine: NlpSqlEngine) -> None:
    result = engine.parse("show high-value customers")

    assert not result.success
    assert result.requires_clarification
    assert result.error is not None
    assert result.error.code == "MISSING_THRESHOLD"
