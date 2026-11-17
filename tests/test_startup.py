"""Bootstrap orchestration checks without network access or a real server."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("startup", Path(__file__).parents[1] / "start.py")
assert spec and spec.loader
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    created = []

    def create(builder, path):
        created.append(path)
        python = path / ("Scripts/python.exe" if startup.os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.touch()

    monkeypatch.setattr(startup.venv.EnvBuilder, "create", create)
    calls = []

    def run(command):
        calls.append(command)
        if "prepare-data" in command:
            Path(command[-1]).write_bytes(b"completed database")

    monkeypatch.setattr(startup, "run", run)
    return tmp_path, calls, created


def test_first_and_repeat_start(checkout):
    root, calls, created = checkout
    assert startup.main([]) == 0
    assert len(created) == 1
    assert len(calls) == 3
    assert calls[0][-1] == "."
    database = root / "data/housing.sqlite"
    assert database.read_bytes() == b"completed database"
    calls.clear()
    assert startup.main([]) == 0
    assert len(calls) == 1
    assert "serve" in calls[0]
    assert len(created) == 1
    assert database.read_bytes() == b"completed database"


def test_dependency_changes_and_options(checkout):
    root, calls, _ = checkout
    assert startup.main(["--setup-only", "--extras", "dev,mcp"]) == 0
    assert calls[0][-1] == ".[dev,mcp]"
    calls.clear()
    (root / "pyproject.toml").write_text("changed")
    assert startup.main(["--extras", "mcp,dev", "--port", "8123", "--host", "localhost"]) == 0
    assert len(calls) == 2
    assert calls[-1][-4:] == ["--host", "localhost", "--port", "8123"]
    calls.clear()
    assert startup.main(["--refresh", "--setup-only"]) == 0
    assert len(calls) == 1


def test_failed_data_build_can_retry(checkout, monkeypatch):
    root, calls, _ = checkout
    original = startup.run

    def fail(command):
        if "prepare-data" in command:
            Path(command[-1]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command)
        original(command)

    monkeypatch.setattr(startup, "run", fail)
    assert startup.main([]) == 1
    assert not (root / "data/housing.sqlite").exists()
    assert list((root / "data").iterdir()) == []
    monkeypatch.setattr(startup, "run", original)
    assert startup.main(["--setup-only"]) == 0


def test_failed_install_invalidates_marker(checkout, monkeypatch):
    root, calls, _ = checkout
    assert startup.main(["--setup-only"]) == 0

    def fail(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(startup, "run", fail)
    assert startup.main(["--refresh"]) == 1
    assert not (root / ".venv/.nlp-sql-setup.json").exists()


def test_custom_existing_database_is_preserved(checkout):
    root, calls, _ = checkout
    database = root / "custom.sqlite"
    database.write_bytes(b"user data")
    assert startup.main(["--db", str(database)]) == 0
    assert len(calls) == 2
    assert database.read_bytes() == b"user data"
    assert str(database) in calls[-1]


def test_invalid_extras_do_not_modify_checkout(checkout):
    root, calls, created = checkout
    with pytest.raises(SystemExit):
        startup.main(["--extras", "unknown"])
    assert not calls and not created
