from nlp_sql.engine import NlpSqlEngine


def test_embedding_parser_orders_expensive_homes_by_price(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("show expensive homes")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY sale_price DESC"
    assert result.matched_rules[0].startswith("embedding:table:housing:")


def test_embedding_parser_orders_many_rooms_by_bedrooms(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("homes with many rooms")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY bedrooms DESC"


def test_embedding_parser_orders_cheap_properties_by_price(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("cheap properties")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY sale_price ASC"


def test_embedding_parser_orders_large_land_properties_by_acreage(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("large land properties")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY acreage DESC"


def test_embedding_parser_orders_recent_house_sales_by_sale_date(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("recent house sales")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY sale_date DESC"


def test_hybrid_fallback_preserves_existing_grouped_having_query() -> None:
    result = NlpSqlEngine().parse("Show customers who spent more than 5 million")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT customer_id, SUM(amount) AS total_purchase FROM orders "
        "GROUP BY customer_id HAVING SUM(amount) > ?"
    )
    assert result.sql.parameters == (5_000_000,)


def test_low_confidence_query_is_rejected() -> None:
    result = NlpSqlEngine().parse("completely unrelated sentence")

    assert not result.success


def test_embedding_path_keeps_write_keywords_rejected(tmp_path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.parse("delete expensive homes")

    assert not result.success
    assert result.error is not None
    assert result.error.code == "FORBIDDEN_OPERATION"
