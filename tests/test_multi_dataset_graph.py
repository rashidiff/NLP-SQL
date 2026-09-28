from pathlib import Path

from nlp_sql.dataset_registry import build_smart_city_demo_registry
from nlp_sql.schema_graph import SchemaGraph


def test_demo_registry_lists_three_datasets(tmp_path: Path) -> None:
    registry = build_smart_city_demo_registry(tmp_path)

    dataset_ids = {dataset.id for dataset in registry.list_datasets()}

    assert dataset_ids == {"properties", "traffic", "facilities"}


def test_schema_graph_expands_alias_to_related_column(tmp_path: Path) -> None:
    registry = build_smart_city_demo_registry(tmp_path)
    graph = SchemaGraph.from_registry(registry)

    alias_key = "traffic/concept/traffic_incidents/near schools"
    expanded = graph.expand((alias_key,), max_depth=1)

    assert any(node.column == "near_facility_id" for node in expanded)
