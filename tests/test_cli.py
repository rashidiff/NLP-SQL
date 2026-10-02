from pathlib import Path

from nlp_sql.cli import main, payload_from_stdout


def test_cli_prepare_data_and_query(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "housing.sqlite"

    assert main(["prepare-data", "--db", str(db_path)]) == 0
    prepared = payload_from_stdout(capsys.readouterr().out)
    assert prepared["database"] == str(db_path)
    assert db_path.exists()

    exit_code = main(
        [
            "query",
            "show top 2 properties by sale price",
            "--db",
            str(db_path),
            "--max-rows",
            "2",
        ]
    )
    payload = payload_from_stdout(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["success"] is True
    assert payload["sql"] == "SELECT * FROM housing ORDER BY sale_price DESC LIMIT ?"
    assert payload["parameters"] == [2]
    assert len(payload["rows"]) == 2


def test_cli_explain_omits_rows(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "housing.sqlite"

    assert main(["explain", "average sale price", "--db", str(db_path)]) == 0
    payload = payload_from_stdout(capsys.readouterr().out)

    assert payload["success"] is True
    assert "rows" not in payload
    assert payload["interpretation"]["source"] == "housing"


def test_cli_parse_with_custom_schema(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        """
        {
          "tickets": {
            "aliases": ["ticket", "tickets"],
            "columns": {
              "id": {"type": "integer", "aliases": ["id", "ticket id"]},
              "priority": {"type": "string", "aliases": ["priority"]},
              "created_at": {"type": "datetime", "aliases": ["created", "created at"]}
            }
          }
        }
        """,
        encoding="utf-8",
    )

    exit_code = main(["parse", "show all tickets", "--schema", str(schema_path)])
    payload = payload_from_stdout(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["success"] is True
    assert payload["ast"]["source"] == "tickets"
    assert payload["sql"] == "SELECT * FROM tickets"
