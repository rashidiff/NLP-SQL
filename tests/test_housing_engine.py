import pytest

from nlp_sql.engine import NlpSqlEngine


@pytest.fixture(scope="module")
def housing_engine(tmp_path_factory: pytest.TempPathFactory) -> NlpSqlEngine:
    db_path = tmp_path_factory.mktemp("housing-db") / "housing.sqlite"
    return NlpSqlEngine.for_housing_dataset(db_path=db_path)


def test_housing_engine_executes_real_database_query(housing_engine: NlpSqlEngine) -> None:
    engine = housing_engine

    result = engine.execute("show top 5 properties by sale price", max_rows=5)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing ORDER BY sale_price DESC LIMIT ?"
    assert result.sql.parameters == (5,)
    assert result.rows is not None
    assert len(result.rows) == 5
    assert "sale_price" in result.rows[0]


def test_housing_engine_filters_real_database_query(housing_engine: NlpSqlEngine) -> None:
    engine = housing_engine

    result = engine.execute("show properties with sale price greater than 500000", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE sale_price > ?"
    assert result.sql.parameters == (500000,)
    assert result.rows is not None
    assert all(row["sale_price"] > 500000 for row in result.rows)


def test_housing_engine_aggregates_real_database_query(housing_engine: NlpSqlEngine) -> None:
    engine = housing_engine

    result = engine.execute("average sale price", max_rows=5)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT AVG(sale_price) AS avg_sale_price FROM housing"
    assert result.rows is not None
    assert result.rows[0]["avg_sale_price"] > 0


def test_housing_engine_filters_year_built(housing_engine: NlpSqlEngine) -> None:
    engine = housing_engine

    result = engine.execute("show properties built after 2000", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE year_built > ?"
    assert result.rows is not None
    assert all(row["year_built"] > 2000 for row in result.rows if row["year_built"] is not None)


def test_housing_engine_filters_string_columns_without_query_templates(
    housing_engine: NlpSqlEngine,
) -> None:
    engine = housing_engine

    result = engine.execute("show properties where land use is single family", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE LOWER(land_use) = LOWER(?)"
    assert result.sql.parameters == ("single family",)
    assert result.rows is not None
    assert {row["land_use"] for row in result.rows} == {"SINGLE FAMILY"}


def test_housing_engine_combines_filters_from_schema_columns(
    housing_engine: NlpSqlEngine,
) -> None:
    engine = housing_engine

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


def test_housing_engine_between_uses_resolved_column(housing_engine: NlpSqlEngine) -> None:
    engine = housing_engine

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


def test_housing_engine_selects_specific_columns_with_filter(
    housing_engine: NlpSqlEngine,
) -> None:
    result = housing_engine.execute(
        "show property address and sale price for properties with sale price above 1000000",
        max_rows=10,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT property_address, sale_price FROM housing WHERE sale_price > ?"
    )
    assert result.sql.parameters == (1_000_000,)
    assert result.rows is not None
    assert result.rows
    assert set(result.rows[0]) == {"property_address", "sale_price"}
    assert all(row["sale_price"] > 1_000_000 for row in result.rows)


def test_housing_engine_counts_with_string_filter(housing_engine: NlpSqlEngine) -> None:
    result = housing_engine.execute("count properties where land use is duplex", max_rows=5)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT COUNT(unique_id) AS count FROM housing WHERE LOWER(land_use) = LOWER(?)"
    )
    assert result.sql.parameters == ("duplex",)
    assert result.rows is not None
    assert result.rows[0]["count"] > 0


def test_housing_engine_aggregates_with_numeric_filter(housing_engine: NlpSqlEngine) -> None:
    result = housing_engine.execute(
        "maximum total value for properties with bedrooms at least 4",
        max_rows=5,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT MAX(total_value) AS max_total_value FROM housing WHERE bedrooms >= ?"
    )
    assert result.sql.parameters == (4,)
    assert result.rows is not None
    assert result.rows[0]["max_total_value"] > 0


def test_housing_engine_sorts_with_multiple_schema_filters(housing_engine: NlpSqlEngine) -> None:
    result = housing_engine.execute(
        "find houses with acreage above 1.5 and full bath at least 2 sort by acreage descending",
        max_rows=10,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT * FROM housing WHERE acreage > ? AND full_bath >= ? ORDER BY acreage DESC"
    )
    assert result.sql.parameters == (1.5, 2)
    assert result.rows is not None
    acreages = [row["acreage"] for row in result.rows]
    assert all(row["acreage"] > 1.5 and row["full_bath"] >= 2 for row in result.rows)
    assert acreages == sorted(acreages, reverse=True)


def test_housing_engine_uses_sale_date_for_natural_date_bounds(
    housing_engine: NlpSqlEngine,
) -> None:
    result = housing_engine.execute("show properties sold before March 1 2013", max_rows=10)

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE sale_date < ?"
    assert result.sql.parameters == ("2013-03-01",)
    assert result.rows is not None
    assert all(row["sale_date"] < "2013-03-01" for row in result.rows)


def test_housing_engine_keeps_injection_like_value_parameterized(
    housing_engine: NlpSqlEngine,
) -> None:
    result = housing_engine.parse("show properties where land use is single family' or 1=1")

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == "SELECT * FROM housing WHERE LOWER(land_use) = LOWER(?)"
    assert result.sql.parameters == ("single family'",)
    assert "1=1" not in result.sql.sql


def test_housing_engine_resolves_unique_id_from_misspelled_address_query(
    housing_engine: NlpSqlEngine,
) -> None:
    result = housing_engine.execute(
        "find the unique id of the hous with this adress : 320  11TH AVE S, NASHVILLE",
        max_rows=20,
    )

    assert result.success
    assert result.sql is not None
    assert result.sql.sql == (
        "SELECT unique_id FROM housing WHERE "
        "LOWER(REPLACE(REPLACE(property_address, ',', ''), '  ', ' ')) = LOWER(?)"
    )
    assert result.sql.parameters == ("320 11th ave s nashville",)
    assert result.rows is not None
    returned_ids = {row["unique_id"] for row in result.rows}
    assert {25631, 25632, 25633}.issubset(returned_ids)
    assert 0 not in returned_ids
