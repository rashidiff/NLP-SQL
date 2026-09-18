"""Command-line interface for preparing data, querying, and serving the API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from nlp_sql.api import run
from nlp_sql.datasets import build_housing_sqlite
from nlp_sql.engine import NlpSqlEngine

DEFAULT_DB_PATH = Path("data/housing.sqlite")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        db_path = build_housing_sqlite(Path(args.db))
        print(json.dumps({"database": str(db_path)}, indent=2))
        return 0

    if args.command == "query":
        engine = NlpSqlEngine.for_housing_dataset(db_path=Path(args.db))
        result = engine.execute(args.query, max_rows=args.max_rows)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.success else 1

    if args.command == "parse":
        engine = NlpSqlEngine.for_housing_dataset(db_path=Path(args.db))
        result = engine.parse(args.query)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.success else 1

    if args.command == "explain":
        engine = NlpSqlEngine.for_housing_dataset(db_path=Path(args.db))
        result = engine.explain(args.query)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.success else 1

    if args.command == "serve":
        run(host=args.host, port=args.port, db_path=Path(args.db))
        return 0

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlp-sql",
        description="Deterministic rule-based NLP-to-SQL over the Kaggle housing database.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-data", help="Download Kaggle data and build SQLite.")
    _add_db_arg(prepare)

    query = subparsers.add_parser("query", help="Parse and execute a natural-language query.")
    query.add_argument("query")
    query.add_argument("--max-rows", type=int, default=20)
    _add_db_arg(query)

    parse = subparsers.add_parser("parse", help="Parse a natural-language query to AST and SQL.")
    parse.add_argument("query")
    _add_db_arg(parse)

    explain = subparsers.add_parser(
        "explain", help="Show semantic interpretation without execution."
    )
    explain.add_argument("query")
    _add_db_arg(explain)

    serve = subparsers.add_parser("serve", help="Run the JSON HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    _add_db_arg(serve)

    return parser


def _add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )


def payload_from_stdout(text: str) -> dict[str, Any]:
    """Test helper for parsing CLI JSON output."""

    return cast(dict[str, Any], json.loads(text))
