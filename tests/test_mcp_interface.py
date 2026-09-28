from nlp_sql.engine import NlpSqlEngine
from nlp_sql.mcp_interface import NlpSqlMcpFacade


def test_mcp_facade_uses_safe_engine_parse() -> None:
    facade = NlpSqlMcpFacade(NlpSqlEngine())

    payload = facade.parse_natural_language("Show customers who spent more than 5 million")

    assert payload["success"] is True
    assert payload["sql"].endswith("HAVING SUM(amount) > ?")


def test_mcp_facade_does_not_execute_write_queries() -> None:
    facade = NlpSqlMcpFacade(NlpSqlEngine())

    payload = facade.execute_query("delete all customers")

    assert payload["success"] is False
    assert payload["error"]["code"] == "FORBIDDEN_OPERATION"
