from pathlib import Path

from nlp_sql.engine import NlpSqlEngine


def test_housing_engine_executes_real_database_query(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute("show top 5 properties by sale price", max_rows=5)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY sale_price DESC LIMIT ?"
    assert result.sql.parameters == (5,)
    assert result.rows is not None
    assert len(result.rows) == 5
    assert "sale_price" in result.rows[0]


def test_housing_engine_filters_real_database_query(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute("show properties with sale price greater than 500000", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE sale_price > ?"
    assert result.sql.parameters == (500000,)
    assert result.rows is not None
    assert all(row["sale_price"] > 500000 for row in result.rows)


def test_housing_engine_aggregates_real_database_query(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute("average sale price", max_rows=5)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT AVG(sale_price) AS avg_sale_price FROM housing"
    assert result.rows is not None
    assert result.rows[0]["avg_sale_price"] > 0


def test_housing_engine_filters_year_built(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute("show properties built after 2000", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE year_built > ?"
    assert result.rows is not None
    assert all(row["year_built"] > 2000 for row in result.rows if row["year_built"] is not None)


def test_housing_engine_filters_string_columns_without_query_templates(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute("show properties where land use is single family", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE LOWER(land_use) = LOWER(?)"
    assert result.sql.parameters == ("single family",)
    assert result.rows is not None
    assert {row["land_use"] for row in result.rows} == {"SINGLE FAMILY"}


def test_housing_engine_combines_filters_from_schema_columns(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute(
        "show properties with bedrooms at least 4 and sale price less than 300000",
        max_rows=10,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE bedrooms >= ? AND sale_price < ?"
    assert result.sql.parameters == (4, 300000)
    assert result.rows is not None
    assert all(row["bedrooms"] >= 4 and row["sale_price"] < 300000 for row in result.rows)


def test_housing_engine_between_uses_resolved_column(tmp_path: Path) -> None:
    engine = NlpSqlEngine.for_housing_dataset(db_path=tmp_path / "housing.sqlite")

    result = engine.execute(
        "show properties with total value between 200000 and 500000",
        max_rows=10,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE total_value BETWEEN ? AND ?"
    assert result.sql.parameters == (200000, 500000)
    assert result.rows is not None
    assert all(200000 <= row["total_value"] <= 500000 for row in result.rows)
