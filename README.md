# NLP-SQL

NLP-SQL is a local Python engine that turns a focused set of English
natural-language questions into validated, parameterized SQL.

The project is designed around the Nashville Housing dataset, but its pipeline
is intentionally split into reusable pieces: normalization, parsing, schema
resolution, AST validation, SQL compilation, safety checks, and optional SQLite
execution.

It is not a generative SQL tool. User text is never interpolated into raw SQL.
The engine first builds a typed `QueryAST`, validates it against a schema
registry, compiles parameterized SQL, and then applies an additional SQL safety
check before execution.

## Features

- Deterministic rule-based parser for supported query patterns.
- Local TF-IDF fallback for flexible schema and intent matching.
- Typed semantic AST as the contract between parsing and SQL generation.
- Schema-aware validation for tables, columns, operators, limits, and
  aggregations.
- Parameterized SQL compilation for SQLite.
- Read-only SQL safety validation.
- CLI commands for data preparation, parsing, explanation, querying, and API
  serving.
- Minimal JSON HTTP API for `/parse`, `/explain`, and `/execute`.
- Automated test coverage for parsing, validation, SQL compilation, CLI, API,
  and dataset import behavior.

## Supported Query Examples

```text
show top 5 properties by sale price
show properties built after 2000
show properties with bedrooms at least 4 and sale price less than 300000
average sale price
show properties where land use is single family
show expensive homes
homes with many rooms
recent house sales
```

## How It Works

```text
Natural-language query
  -> text normalization
  -> rule parser or local TF-IDF fallback
  -> schema resolution
  -> typed QueryAST
  -> AST validation
  -> parameterized SQL compilation
  -> SQL safety validation
  -> optional SQLite execution
```

The rule parser handles explicit supported grammar. The TF-IDF fallback helps
with looser phrases such as "expensive homes" or "recent house sales" while
still staying local, schema-bound, and non-generative.

## Installation

```bash
git clone https://github.com/rashidiff/NLP-SQL.git
cd NLP-SQL
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Prepare The Dataset

Build the local SQLite database from the configured Kaggle housing dataset:

```bash
nlp-sql prepare-data
```

By default this creates:

```text
data/housing.sqlite
```

You can choose a different database path:

```bash
nlp-sql prepare-data --db ./tmp/housing.sqlite
```

## CLI Usage

Parse a question and show the generated AST and SQL:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
```

Explain the semantic interpretation without executing SQL:

```bash
nlp-sql explain "average sale price"
```

Execute a query against the SQLite database:

```bash
nlp-sql query "show top 5 properties by sale price"
```

Limit returned rows:

```bash
nlp-sql query "show expensive homes" --max-rows 10
```

## HTTP API

Start the local API server:

```bash
nlp-sql serve
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"query":"show top 5 properties by sale price"}'
```

Available endpoints:

- `POST /parse`
- `POST /explain`
- `POST /execute`

## Development

Run the test suite:

```bash
python -m pytest
```

Run linting and type checks:

```bash
python -m ruff check .
python -m mypy
```

## Project Layout

```text
src/nlp_sql/
  api.py              HTTP request handling
  cli.py              command-line interface
  compiler.py         QueryAST to parameterized SQL
  datasets.py         housing dataset import and SQLite setup
  embedding_parser.py local TF-IDF fallback parser
  engine.py           public orchestration API
  parser.py           deterministic rule-based parser
  query_ast.py        typed semantic query model
  safety.py           input and SQL safety checks
  schema.py           schema registry and aliases
  validator.py        AST validation
tests/                regression and integration tests
```

For a deeper design discussion, see `ARCHITECTURE.md` and
`SYSTEM_DESIGN.md`.

## Current Scope

NLP-SQL is intentionally narrow. It is best suited for read-only analytical
queries over known schemas. It does not attempt to support arbitrary SQL,
multi-turn conversational repair, joins across unknown schemas, or writes such
as `INSERT`, `UPDATE`, and `DELETE`.
