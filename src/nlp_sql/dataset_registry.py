"""Dataset registry for multi-source semantic query planning."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from nlp_sql.executor import SQLiteExecutor
from nlp_sql.retrieval import HybridSchemaRetriever, documents_from_schema
from nlp_sql.schema import SchemaRegistry


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    title: str
    description: str
    source: str
    schema: SchemaRegistry
    db_path: Path
    semantic_metadata: dict[str, str] = field(default_factory=dict)

    def executor(self) -> SQLiteExecutor:
        return SQLiteExecutor(self.db_path)

    def retriever(self) -> HybridSchemaRetriever:
        return HybridSchemaRetriever.with_sqlite_store(
            documents_from_schema(
                self.schema,
                dataset_id=self.id,
                dataset_title=self.title,
                dataset_description=self.description,
            ),
            self.db_path.with_suffix(".embeddings.sqlite"),
        )


class DatasetRegistry:
    """Registry of local datasets and their schema/execution metadata."""

    def __init__(self, datasets: tuple[DatasetInfo, ...]) -> None:
        self._datasets = {dataset.id: dataset for dataset in datasets}

    def list_datasets(self) -> tuple[DatasetInfo, ...]:
        return tuple(self._datasets.values())

    def get(self, dataset_id: str) -> DatasetInfo | None:
        return self._datasets.get(dataset_id)

    def search_datasets(self, query: str, top_k: int = 3) -> tuple[dict[str, object], ...]:
        candidates: list[dict[str, object]] = []
        text = query.lower()
        for dataset in self._datasets.values():
            haystack = " ".join(
                (
                    dataset.id,
                    dataset.title,
                    dataset.description,
                    dataset.source,
                    *dataset.semantic_metadata.values(),
                )
            ).lower()
            score = sum(1 for token in text.split() if token in haystack) / max(
                len(text.split()), 1
            )
            if score > 0:
                candidates.append(
                    {
                        "dataset_id": dataset.id,
                        "title": dataset.title,
                        "description": dataset.description,
                        "score": score,
                    }
                )
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        return tuple(candidates[:top_k])


def build_smart_city_demo_registry(root: Path) -> DatasetRegistry:
    """Create a small local smart-city/open-data registry for demos and tests."""

    root.mkdir(parents=True, exist_ok=True)
    datasets = (
        _build_property_dataset(root / "properties.sqlite"),
        _build_traffic_dataset(root / "traffic.sqlite"),
        _build_facilities_dataset(root / "facilities.sqlite"),
    )
    return DatasetRegistry(datasets)


def _build_property_dataset(db_path: Path) -> DatasetInfo:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS properties (
                property_id INTEGER PRIMARY KEY,
                area TEXT,
                land_use TEXT,
                sale_price INTEGER,
                acreage REAL
            )
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO properties
            (property_id, area, land_use, sale_price, acreage)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "north", "residential", 325000, 0.25),
                (2, "east", "residential", 280000, 0.15),
                (3, "central", "commercial", 900000, 0.40),
            ],
        )
        connection.commit()
    schema = SchemaRegistry.from_sqlite(
        db_path,
        table_aliases={"properties": ("housing", "homes", "houses", "property")},
        column_aliases={"acreage": ("lot size", "large lots", "land size")},
    )
    return DatasetInfo(
        id="properties",
        title="Property Records",
        description="Synthetic property sales and land-use records.",
        source="local synthetic fixture",
        schema=schema,
        db_path=db_path,
        semantic_metadata={"domain": "housing property land use sale price acreage"},
    )


def _build_traffic_dataset(db_path: Path) -> DatasetInfo:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_incidents (
                incident_id INTEGER PRIMARY KEY,
                area TEXT,
                incident_type TEXT,
                severity INTEGER,
                occurred_at TEXT,
                near_facility_id INTEGER
            )
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO traffic_incidents
            (incident_id, area, incident_type, severity, occurred_at, near_facility_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "north", "collision", 3, "2026-01-10", 1),
                (2, "north", "speeding", 1, "2026-01-12", 1),
                (3, "east", "collision", 2, "2026-02-01", 2),
            ],
        )
        connection.commit()
    schema = SchemaRegistry.from_sqlite(
        db_path,
        table_aliases={"traffic_incidents": ("traffic", "incidents", "accidents")},
        column_aliases={
            "near_facility_id": ("near schools", "near facilities", "facility"),
            "severity": ("serious", "severity"),
        },
    )
    return DatasetInfo(
        id="traffic",
        title="Traffic Incidents",
        description="Synthetic traffic incident records by city area.",
        source="local synthetic fixture",
        schema=schema,
        db_path=db_path,
        semantic_metadata={"domain": "traffic incidents accidents collision schools areas"},
    )


def _build_facilities_dataset(db_path: Path) -> DatasetInfo:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS public_facilities (
                facility_id INTEGER PRIMARY KEY,
                name TEXT,
                facility_type TEXT,
                area TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO public_facilities
            (facility_id, name, facility_type, area)
            VALUES (?, ?, ?, ?)
            """,
            [
                (1, "North Elementary", "school", "north"),
                (2, "East Library", "library", "east"),
                (3, "Central High", "school", "central"),
            ],
        )
        connection.commit()
    schema = SchemaRegistry.from_sqlite(
        db_path,
        table_aliases={"public_facilities": ("facilities", "schools", "public buildings")},
        column_aliases={"facility_type": ("type", "school", "library")},
    )
    return DatasetInfo(
        id="facilities",
        title="Public Facilities",
        description="Synthetic public facility locations and categories.",
        source="local synthetic fixture",
        schema=schema,
        db_path=db_path,
        semantic_metadata={"domain": "schools libraries public facilities city services"},
    )
