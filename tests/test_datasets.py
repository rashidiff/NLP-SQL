import sqlite3
from pathlib import Path

from nlp_sql.datasets import build_housing_sqlite
from nlp_sql.schema import SchemaRegistry


def test_build_housing_sqlite_imports_kaggle_csv(tmp_path: Path) -> None:
    db_path = build_housing_sqlite(tmp_path / "housing.sqlite")

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM housing").fetchone()[0]
        row = connection.execute(
            "SELECT sale_date, sale_price, year_built FROM housing "
            "WHERE year_built IS NOT NULL ORDER BY unique_id LIMIT 1"
        ).fetchone()

    assert count > 0
    assert row[0].count("-") == 2
    assert isinstance(row[1], int)
    assert isinstance(row[2], int)


def test_schema_registry_can_introspect_housing_sqlite(tmp_path: Path) -> None:
    db_path = build_housing_sqlite(tmp_path / "housing.sqlite")

    registry = SchemaRegistry.from_sqlite(
        db_path,
        table_aliases={"housing": ("properties",)},
        column_aliases={"sale_price": ("price",)},
    )

    assert registry.resolve_table("properties").canonical == "housing"  # type: ignore[union-attr]
    assert registry.resolve_column("price", table="housing")[0].column == "sale_price"
    assert registry.get_column("housing", "sale_date").type == "datetime"  # type: ignore[union-attr]
