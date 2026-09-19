"""Dataset loaders for local, real-database development."""

from __future__ import annotations

import csv
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import kagglehub  # type: ignore[import-untyped]

HOUSING_DATASET = "bvanntruong/housing-sql-project"
HOUSING_CSV_NAME = "Nashville Housing.csv"
HOUSING_TABLE = "housing"


HOUSING_TABLE_ALIASES = {
    HOUSING_TABLE: (
        "housing",
        "house",
        "hous",
        "houses",
        "property",
        "properties",
        "home",
        "homes",
        "nashville housing",
        "sales",
    )
}

HOUSING_COLUMN_ALIASES = {
    "unique_id": ("unique id", "id"),
    "parcel_id": ("parcel id", "parcel"),
    "land_use": ("land use", "property type", "type", "category"),
    "property_address": ("property address", "address", "adress"),
    "sale_date": ("sale date", "date", "sold date", "sale time"),
    "sale_price": ("sale price", "price", "selling price", "sold price", "amount"),
    "legal_reference": ("legal reference", "reference"),
    "sold_as_vacant": ("sold as vacant", "vacant"),
    "owner_name": ("owner name", "owner"),
    "owner_address": ("owner address",),
    "acreage": ("acreage", "acres", "land size"),
    "tax_district": ("tax district", "district"),
    "land_value": ("land value",),
    "building_value": ("building value",),
    "total_value": ("total value", "value", "assessed value"),
    "year_built": ("year built", "built", "built year"),
    "bedrooms": ("bedrooms", "beds"),
    "full_bath": ("full bath", "bathrooms", "full bathrooms"),
    "half_bath": ("half bath", "half bathrooms"),
}


def download_housing_dataset() -> Path:
    """Download the Kaggle housing dataset and return its local directory."""

    return Path(kagglehub.dataset_download(HOUSING_DATASET))


def build_housing_sqlite(db_path: Path, dataset_path: Path | None = None) -> Path:
    """Create a SQLite database from the Kaggle Nashville Housing CSV."""

    source_dir = dataset_path or download_housing_dataset()
    csv_path = source_dir / HOUSING_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {HOUSING_CSV_NAME} in {source_dir}.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as connection:
        connection.execute(_create_housing_table_sql())
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = [_clean_housing_row(row) for row in reader]
        placeholders = ", ".join("?" for _ in _HOUSING_COLUMNS)
        columns = ", ".join(_HOUSING_COLUMNS)
        connection.executemany(
            f"INSERT INTO {HOUSING_TABLE} ({columns}) VALUES ({placeholders})",
            [[row[column] for column in _HOUSING_COLUMNS] for row in rows],
        )
        connection.commit()
    return db_path


def _create_housing_table_sql() -> str:
    return f"""
    CREATE TABLE {HOUSING_TABLE} (
        unique_id INTEGER PRIMARY KEY,
        parcel_id TEXT,
        land_use TEXT,
        property_address TEXT,
        sale_date TEXT,
        sale_price INTEGER,
        legal_reference TEXT,
        sold_as_vacant TEXT,
        owner_name TEXT,
        owner_address TEXT,
        acreage REAL,
        tax_district TEXT,
        land_value INTEGER,
        building_value INTEGER,
        total_value INTEGER,
        year_built INTEGER,
        bedrooms INTEGER,
        full_bath INTEGER,
        half_bath INTEGER
    )
    """


_HOUSING_COLUMNS = (
    "unique_id",
    "parcel_id",
    "land_use",
    "property_address",
    "sale_date",
    "sale_price",
    "legal_reference",
    "sold_as_vacant",
    "owner_name",
    "owner_address",
    "acreage",
    "tax_district",
    "land_value",
    "building_value",
    "total_value",
    "year_built",
    "bedrooms",
    "full_bath",
    "half_bath",
)


def _clean_housing_row(row: dict[str, str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for raw_key, value in row.items():
        key = _snake_case(raw_key.strip())
        cleaned[key] = _clean_value(key, value)
    return cleaned


def _clean_value(key: str, value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    if key == "sale_date":
        return datetime.strptime(text, "%B %d, %Y").date().isoformat()
    if key in {
        "unique_id",
        "sale_price",
        "land_value",
        "building_value",
        "total_value",
        "year_built",
        "bedrooms",
        "full_bath",
        "half_bath",
    }:
        return int(float(text.replace(",", "").replace("$", "")))
    if key == "acreage":
        return float(text)
    return text


def _snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()
