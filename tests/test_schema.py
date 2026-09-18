from nlp_sql.schema import SchemaRegistry


def test_default_schema_resolves_table_aliases() -> None:
    registry = SchemaRegistry.default()

    assert registry.resolve_table("clients").canonical == "customers"  # type: ignore[union-attr]
    assert registry.resolve_table("purchases").canonical == "orders"  # type: ignore[union-attr]


def test_default_schema_resolves_column_alias_with_table() -> None:
    registry = SchemaRegistry.default()

    columns = registry.resolve_column("revenue", table="orders")

    assert len(columns) == 1
    assert columns[0].table == "orders"
    assert columns[0].column == "amount"
    assert columns[0].type == "decimal"


def test_default_schema_returns_ambiguous_column_candidates_without_table() -> None:
    registry = SchemaRegistry.default()

    columns = registry.resolve_column("id")

    assert {column.table for column in columns} == {"customers", "orders"}


def test_schema_whitelists_tables_and_columns() -> None:
    registry = SchemaRegistry.default()

    assert registry.has_table("orders")
    assert registry.has_column("orders", "amount")
    assert not registry.has_table("users")
    assert not registry.has_column("orders", "password")
