# NLP-SQL

## Project Description

NLP-SQL is a deterministic, rule-based system that converts English natural
language questions into safe SQL queries. It does not use LLMs, generative AI,
embeddings, vector databases, or external AI services.

The project is designed to work with the real Nashville Housing dataset. Users
can ask questions such as:

```text
show top 5 properties by sale price
show properties with bedrooms at least 4 and sale price less than 300000
average sale price
show properties where land use is single family
```

The system converts the query into a typed semantic AST, validates that AST
against the database schema, compiles parameterized SQL, and optionally executes
the query against SQLite.

## System Overview

Main runtime flow:

```text
Natural Language Query
        ↓
Text Normalizer
        ↓
Tokenizer / Phrase Matcher
        ↓
Rule-Based Semantic Parser
        ↓
Schema Resolver
        ↓
Typed Query AST
        ↓
AST Validator
        ↓
Parameterized SQL Compiler
        ↓
SQL Safety Validator
        ↓
SQLite Executor
        ↓
Rows
```

Real database setup flow:

```text
Kaggle CSV Dataset
        ↓
SQLite Importer
        ↓
SQLite Schema Introspection
        ↓
Schema Registry
        ↓
NLP-to-AST
        ↓
Validated SQL
```

Important rules:

- The parser never generates raw SQL directly from user text.
- The central contract is `QueryAST`.
- Tables, columns, operators, and aggregations are whitelisted and validated.
- User values are passed as SQL parameters, not interpolated into SQL strings.
- The HTTP API is optional and is only a wrapper around the deterministic parser.
  It is not an AI model or AI service.

## Repository Tree

```text
NLP-SQL/
├── ARCHITECTURE.md
├── README.md
├── SYSTEM_DESIGN.md
├── pyproject.toml
├── src/
│   └── nlp_sql/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── cli.py
│       ├── compiler.py
│       ├── datasets.py
│       ├── date_parser.py
│       ├── engine.py
│       ├── executor.py
│       ├── normalizer.py
│       ├── number_parser.py
│       ├── parser.py
│       ├── query_ast.py
│       ├── result.py
│       ├── safety.py
│       ├── schema.py
│       ├── tokenizer.py
│       ├── validator.py
│       └── config/
│           ├── __init__.py
│           └── default_schema.json
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_cli.py
    ├── test_compiler_validator_security.py
    ├── test_datasets.py
    ├── test_date_parser.py
    ├── test_engine_integration.py
    ├── test_housing_engine.py
    ├── test_normalizer.py
    ├── test_number_parser.py
    ├── test_package.py
    ├── test_schema.py
    └── test_tokenizer.py
```

## Clone And Run

Clone the repository:

```bash
git clone https://github.com/rashidiff/NLP-SQL.git
cd NLP-SQL
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Activate it on macOS / Linux:

```bash
source .venv/bin/activate
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

Run a real natural-language query against the database:

```bash
nlp-sql query "show top 5 properties by sale price"
```

More query examples:

```bash
nlp-sql query "show properties built after 2000"
nlp-sql query "show properties with bedrooms at least 4 and sale price less than 300000"
nlp-sql query "average sale price"
nlp-sql query "show properties where land use is single family"
```

Show the generated AST and SQL without executing:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
```

Show the semantic interpretation:

```bash
nlp-sql explain "average sale price"
```

Run the optional HTTP API:

```bash
nlp-sql serve
```

Run tests and checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```
