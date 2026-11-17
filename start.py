"""Bootstrap a checkout and serve NLP-SQL using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=Path("data/housing.sqlite"))
    parser.add_argument("--extras", default="", help="Comma-separated extras: dev,semantic,mcp")
    parser.add_argument("--refresh", action="store_true", help="Reinstall dependencies")
    parser.add_argument(
        "--setup-only", action="store_true", help="Prepare without starting the API"
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):  # noqa: UP036 - runs before package installation
        parser.error("Python 3.11 or newer is required.")
    extras = sorted(set(filter(None, (item.strip() for item in args.extras.split(",")))))
    if set(extras) - {"dev", "semantic", "mcp"}:
        parser.error("Supported extras are dev, semantic, and mcp.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / ".nlp-sql-setup.json"
    try:
        if not python.exists():
            print("Creating .venv...", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        signature = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
        state = {"project": signature, "extras": extras, "root": str(ROOT)}
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            installed = None
        if args.refresh or installed != state:
            # An interrupted reinstall must be retried, even if an older marker exists.
            marker.unlink(missing_ok=True)
            package = "." + (f"[{','.join(extras)}]" if extras else "")
            print("Installing dependencies...", flush=True)
            run([str(python), "-m", "pip", "install", "-e", package])
            marker.write_text(json.dumps(state), encoding="utf-8")
        database = args.db if args.db.is_absolute() else ROOT / args.db
        if not database.exists():
            database.parent.mkdir(parents=True, exist_ok=True)
            # Publish only a completed database; retries never reuse a partial build.
            with tempfile.TemporaryDirectory(dir=database.parent) as staging:
                prepared = Path(staging) / "housing.sqlite"
                print("Preparing housing database...", flush=True)
                run([str(python), "-m", "nlp_sql", "prepare-data", "--db", str(prepared)])
                prepared.replace(database)
        if not args.setup_only:
            print(
                f"Starting NLP-SQL at http://{args.host}:{args.port} "
                "(POST /parse, /explain, /execute). Ctrl+C stops the server.",
                flush=True,
            )
            run(
                [
                    str(python),
                    "-m",
                    "nlp_sql",
                    "serve",
                    "--db",
                    str(database),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ]
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"Startup failed: {error}\nFix the reported error and rerun the same command. "
            "Dependency installation and the first dataset download require network access.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
