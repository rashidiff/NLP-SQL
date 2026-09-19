# System Design

This document explains how the NLP-SQL system works, what technologies it uses,
and the main engineering ideas behind the implementation.

## Goal

The system converts English natural-language questions into safe SQL queries for
the Nashville Housing SQLite database.

Example:

```text
show top 5 properties by sale price
```

becomes:

```sql
SELECT * FROM housing ORDER BY sale_price DESC LIMIT ?
```

with parameters:

```text
[5]
```

## High-Level Flow

```text
English query
    ↓
Text normalization
    ↓
Tokenization and phrase matching
    ↓
Schema-aware rule-based parsing
    ↓
Typed Query AST
    ↓
AST validation
    ↓
Parameterized SQL compilation
    ↓
SQL safety validation
    ↓
SQLite execution
```

## Technologies Used

- Python 3.11
- SQLite
- KaggleHub, for downloading the Nashville Housing CSV dataset
- Python standard library modules:
  - `csv`
  - `sqlite3`
  - `argparse`
  - `http.server`
  - `dataclasses`
  - `re`
  - `datetime`
- Pytest for tests
- Ruff for formatting and linting
- Mypy for static type checking

## Main Ideas

### 1. Schema-Aware Parsing

The parser does not try to support arbitrary English by guessing. It resolves
terms against the actual database schema.

Examples:

```text
property / properties / house / home → housing
sale price / price / sold price       → sale_price
land use / property type              → land_use
year built / built                    → year_built
```

The SQLite schema is introspected, then enriched with deterministic aliases.

### 2. Query AST As The Core Contract

The parser does not directly produce SQL. It produces a typed `QueryAST`.

This makes the pipeline safer:

```text
Natural language
    ↓
QueryAST
    ↓
Validation
    ↓
SQL
```

The SQL compiler only accepts structured AST nodes.

### 3. Parameterized SQL

User-provided values are never interpolated into SQL strings.

For example:

```text
show properties where land use is single family
```

compiles to:

```sql
SELECT * FROM housing WHERE LOWER(land_use) = LOWER(?)
```

with:

```text
["single family"]
```

### 4. Deterministic Rules

The parser uses deterministic rules for:

- table/entity resolution
- column resolution
- operators
- numbers
- aggregations
- sorting
- limits
- date ranges

Examples:

```text
greater than → >
less than    → <
at least     → >=
between      → BETWEEN
top 5        → LIMIT 5 + descending sort
```

### 5. Real Database Execution

The dataset loader downloads the Kaggle CSV and imports it into SQLite:

```text
Kaggle CSV
    ↓
SQLite table: housing
```

The engine can then execute generated SQL through `SQLiteExecutor`.

### 6. CLI-First Usage

The primary interface is the CLI:

```bash
nlp-sql prepare-data
nlp-sql query "show top 5 properties by sale price"
nlp-sql parse "show properties with bedrooms at least 4"
nlp-sql explain "average sale price"
```

The HTTP API is optional and wraps the same engine.

## Important Modules

```text
src/nlp_sql/normalizer.py      text normalization
src/nlp_sql/number_parser.py   English and formatted number parsing
src/nlp_sql/tokenizer.py       tokenization and phrase matching
src/nlp_sql/schema.py          schema registry and SQLite introspection
src/nlp_sql/parser.py          rule-based semantic parser
src/nlp_sql/query_ast.py       typed AST nodes
src/nlp_sql/validator.py       AST validation
src/nlp_sql/compiler.py        AST to parameterized SQL
src/nlp_sql/safety.py          read-only and SQL safety checks
src/nlp_sql/datasets.py        Kaggle CSV to SQLite loader
src/nlp_sql/executor.py        SQLite query execution
src/nlp_sql/engine.py          orchestration layer
src/nlp_sql/cli.py             command-line interface
```

## Example End-To-End

Command:

```bash
nlp-sql query "show properties with bedrooms at least 4 and sale price less than 300000"
```

Interpretation:

```text
source: housing
filters:
  bedrooms >= 4
  sale_price < 300000
```

SQL:

```sql
SELECT * FROM housing WHERE bedrooms >= ? AND sale_price < ?
```

Parameters:

```text
[4, 300000]
```

