# NLP-SQL

NLP-SQL is a deterministic, rule-based Natural Language to SQL engine. It does
not use LLMs, embeddings, vector databases, generative AI, or external AI
services.

English business queries are normalized, tokenized, parsed into a typed Query
AST, validated against a schema registry, and compiled into parameterized SQL.

## Quick Start

From a fresh clone:

```bash
git clone https://github.com/rashidiff/NLP-SQL.git
cd NLP-SQL
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Download the Kaggle dataset and build the local SQLite database:

```bash
nlp-sql prepare-data
```

Run a real natural-language query against the real SQLite database:

```bash
nlp-sql query "show top 5 properties by sale price"
```

Other useful commands:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
nlp-sql explain "average sale price"
nlp-sql serve
```

The default database path is `data/housing.sqlite`. To use another location:

```bash
nlp-sql prepare-data --db C:\temp\housing.sqlite
nlp-sql query "show properties built after 2000" --db C:\temp\housing.sqlite
```

## Project Overview

The first version supports a deliberately small grammar for customer/order
analytics:

- Selecting configured tables.
- Filtering numeric metrics such as order amount and revenue.
- Counting, summing, averaging, minimum, and maximum aggregations.
- Grouping by customer and month-oriented fields.
- Date ranges such as today, yesterday, last month, last 7 days, and explicit
  month/day/year bounds.
- Sorting and limits.
- Explain mode for deterministic interpretation details.

Unsupported or ambiguous requests are rejected instead of guessed.

The project can also run against the real Kaggle Nashville Housing dataset:

```python
import kagglehub

path = kagglehub.dataset_download("bvanntruong/housing-sql-project")
print("Path to dataset files:", path)
```

`NlpSqlEngine.for_housing_dataset()` downloads that dataset, imports
`Nashville Housing.csv` into a local SQLite database, introspects the SQLite
schema, and uses that real schema for parsing, validation, SQL compilation, and
execution.

## Architecture

```text
Natural Language Query
        ↓
Text Normalizer
        ↓
Tokenizer / Phrase Matcher
        ↓
Intent Detector / Entity Extractor
        ↓
Schema Resolver
        ↓
Semantic Query AST
        ↓
AST Validator
        ↓
SQL Compiler
        ↓
SQL Safety Validator
        ↓
Optional Database Executor
```

The parser never emits raw SQL. The central contract is
`nlp_sql.query_ast.QueryAST`, which lets a future parser implementation produce
the same semantic representation without changing validation or SQL compilation.

## Installation

```bash
python -m pip install -e ".[dev]"
```

The runtime uses the Python standard library plus `kagglehub` for downloading
the Kaggle dataset.

## Configuration

Schema and aliases live in:

```text
src/nlp_sql/config/default_schema.json
```

This file defines whitelisted tables, columns, types, and aliases. Parser logic
resolves table and column phrases through `SchemaRegistry`; schema aliases are
not scattered through controllers or SQL compilation.

## Running the Application

```bash
nlp-sql serve
```

The server listens on `127.0.0.1:8000`. By default, the command-line server
initializes the Kaggle housing dataset at `data/housing.sqlite`.

## Running Tests

```bash
python -m ruff format .
python -m ruff check .
python -m mypy
python -m pytest
```

## API Documentation

### `POST /parse`

Request:

```json
{
  "query": "Show customers who spent more than 5 million"
}
```

Response:

```json
{
  "success": true,
  "ast": {},
  "sql": "SELECT customer_id, SUM(amount) AS total_purchase FROM orders GROUP BY customer_id HAVING SUM(amount) > ?",
  "parameters": [5000000]
}
```

### `POST /explain`

Request:

```json
{
  "query": "Show the top 5 customers by total spending this month"
}
```

Response includes `interpretation` and `matched_rules`, but no SQL execution.

### `POST /execute`

Runs a parsed, validated, parameterized read-only query against the configured
SQLite database.

Request:

```json
{
  "query": "show top 5 properties by sale price",
  "max_rows": 5
}
```

Example response fields:

```json
{
  "success": true,
  "sql": "SELECT * FROM housing ORDER BY sale_price DESC LIMIT ?",
  "parameters": [5],
  "rows": [
    {
      "unique_id": 24392,
      "sale_price": 50000000
    }
  ]
}
```

## Supported Natural Language Queries

Examples covered by tests include:

- `Show all customers`
- `Show all orders`
- `Show orders above 5 million`
- `Show orders between 100 and 500`
- `How many orders were placed today?`
- `What is the total revenue this month?`
- `Show customers who spent more than 10000`
- `Show the 5 customers with the highest total spending`
- `Show the latest 20 orders`
- `Sort customers by name ascending`
- `Count orders per customer`
- `Show orders after January 1 2026`
- `Show orders between January 1 2026 and January 31 2026`
- `show top 5 properties by sale price`
- `show properties with sale price greater than 500000`
- `average sale price`
- `show properties built after 2000`

## Query AST

The AST is strongly typed with dataclasses:

- `QueryAST`
- `ColumnExpression`
- `AggregationExpression`
- `Predicate`
- `DateRangePredicate`
- `HavingPredicate`
- `OrderBy`

Values are stored as data, not SQL fragments.

## Schema Registry

`SchemaRegistry` loads configured table and column metadata, resolves aliases,
and provides whitelist checks to validators and parsers.

For real databases, `SchemaRegistry.from_sqlite(...)` introspects SQLite tables
and columns, then merges deterministic table/column aliases such as
`properties → housing` and `sale price → sale_price`.

## Adding Tables

Add a table entry to `default_schema.json`:

```json
{
  "products": {
    "aliases": ["product", "products"],
    "columns": {
      "id": {"type": "integer"},
      "name": {"type": "string"}
    }
  }
}
```

## Adding Columns

Add the column under the table with a type and aliases:

```json
"amount": {
  "type": "decimal",
  "aliases": ["revenue", "sales amount", "order amount"]
}
```

## Adding Synonyms

Add aliases in schema metadata. For example, to support `turnover` as revenue,
add it to `orders.columns.amount.aliases`.

## Adding Operators

Phrase-level operators are configured in `tokenizer.DEFAULT_PHRASE_RULES`.
Single-word amount operators are in `RuleBasedParser._OPERATOR_WORDS`.

## Adding Aggregations

Aggregation phrases are centralized in `RuleBasedParser._AGGREGATION_RULES` and
validated against `SUPPORTED_AGGREGATIONS`.

## Adding Date Expressions

Add deterministic phrase handling in `DateExpressionParser`. Date parsing
resolves to explicit start/end dates before SQL compilation.

## Adding SQL Dialects

Create a new `SqlDialect` subclass and pass it to `SqlCompiler`. The AST and
validator remain unchanged.

## Security Model

- Read-only queries only.
- Forbidden write/DDL keywords are rejected.
- Tables and columns must exist in the schema registry.
- Operators and aggregations are whitelisted.
- User values are parameterized with `?` placeholders.
- The compiler only accepts validated AST nodes.
- The generated SQL is checked for forbidden keywords and semicolons.
- Database execution is isolated in `SQLiteExecutor`; parsing and compilation
  remain testable without a database connection.

## Known Limitations

- This is not a general English parser.
- Joins are not yet modeled.
- Grouping by month currently groups by the configured date column directly.
- The bundled static demo schema only includes `customers` and `orders`.
- Real execution currently targets SQLite.

## Roadmap

- Add explicit join AST nodes.
- Add richer date bucketing such as `DATE_TRUNC`.
- Add more dialects.
- Add configurable parser-rule files.
- Add optional database execution behind the existing parse/compile boundary.
