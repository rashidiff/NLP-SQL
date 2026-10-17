# NLP-SQL

NLP-SQL is a schema-aware, safety-oriented natural-language query engine for
turning natural-language questions into validated, parameterized SQL.

The central contract is still `QueryAST`. Natural language never becomes SQL
directly. Every query must pass through structured parsing, AST validation,
parameterized compilation, and SQL safety validation before execution.

The project has evolved into a hybrid semantic query system:

- deterministic rule-based parsing as a fast fallback;
- schema retrieval over datasets, tables, columns, aliases, descriptions,
  relational metadata, profiles, and representative values;
- local embedding retrieval for schema matching only, never SQL generation;
- adaptive vector search telemetry and interpretable search-budget policy;
- SQLite-backed embedding storage with a bounded in-memory LRU cache;
- ambiguity-aware Text-to-SQL evaluation utilities;
- multi-dataset registry and lightweight schema graph expansion;
- registry-aware dataset routing with ambiguity clarification;
- optional MCP-style facade that wraps the same safe engine.

## What It Is Not

NLP-SQL is not an LLM-to-SQL demo and does not ask a model to write SQL. The
optional embedding layer is used only to retrieve relevant schema concepts. SQL
is generated only from validated `QueryAST` objects.

The research-inspired modules are technically honest approximations. They are
inspired by systems and evaluation ideas, but they do not claim to reproduce the
full algorithms from those papers. See `PAPER_MAPPING.md`.

## Architecture

```text
Natural-language query
  -> dataset/schema retrieval
       -> lexical retrieval
       -> local embedding retrieval
       -> hybrid ranking
       -> optional schema graph expansion
  -> LLM semantic planner or deterministic fallback
  -> typed QueryAST and boolean filter tree
  -> AST validation
  -> parameterized SQL compilation
  -> SQL safety validation
  -> EXPLAIN and guarded read-only SQLite execution
```

Retrieval and telemetry can inform explanations and future adaptive ranking, but
they must not bypass validation.

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

Install the core package:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional local sentence-transformers support:

```bash
python -m pip install -e ".[semantic]"
```

The default semantic backend is a deterministic local hash embedding backend, so
the project works without downloading a transformer model.

Optional LLM semantic parsing can be enabled with an OpenAI-compatible chat
completion endpoint. The model returns a structured query plan, never SQL:

```bash
# PowerShell
$env:NLP_SQL_LLM_API_KEY = "..."
$env:NLP_SQL_LLM_MODEL = "gpt-4o-mini"
$env:NLP_SQL_LLM_BASE_URL = "https://api.openai.com/v1"
nlp-sql parse "show expensive homes with at least four bedrooms"
```

`NLP_SQL_LLM_BASE_URL` may point to a compatible local or hosted endpoint. The
existing deterministic parser remains the fallback when the LLM is disabled,
unavailable, ambiguous, or produces an invalid plan. Both paths use the same
AST validator, parameterized compiler, and SQL safety checks.

When a SQLite housing database is opened, the engine profiles whitelisted
columns (null ratio, distinct count, min/max, and sample values) and attaches
that evidence to schema retrieval documents. Execution results can include the
SQLite query plan under `execution_plan`.

For multiple registered datasets, `RegistryQueryEngine` first selects a
dataset using registry metadata and routes the query to that dataset's schema,
retriever, LLM planner, and executor. Close dataset scores result in an
explicit clarification instead of a silent cross-dataset guess.

## CLI Usage

Prepare the Nashville Housing SQLite database:

```bash
nlp-sql prepare-data
```

Parse a query and show generated AST/SQL:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
```

Parse against your own schema JSON without preparing a demo database:

```bash
nlp-sql parse "show all tickets" --schema ./schema.json
```

Explain without execution:

```bash
nlp-sql explain "show expensive homes"
```

Execute against SQLite:

```bash
nlp-sql query "show top 5 properties by sale price"
```

Run the HTTP API:

```bash
nlp-sql serve
```

## Programmatic Examples

Schema retrieval without SQL generation:

```python
from nlp_sql.engine import NlpSqlEngine

engine = NlpSqlEngine.for_housing_dataset()
print(engine.explain_schema_match("show houses with large lots"))
```

Safe parse with retrieval evidence and adaptive decision:

```python
result = engine.parse("show expensive homes")
print(result.to_dict()["schema_candidates"])
print(result.to_dict()["adaptive_decision"])
```

Smart-city demo registry:

```python
from pathlib import Path
from nlp_sql.dataset_registry import build_smart_city_demo_registry
from nlp_sql.schema_graph import SchemaGraph

registry = build_smart_city_demo_registry(Path("data/demo"))
graph = SchemaGraph.from_registry(registry)
```

## Current Safety Properties

- Only structured `QueryAST` objects are compiled.
- Tables, columns, operators, aggregations, ordering, and limits are validated.
- Join tables and join keys are validated against the configured schema.
- User values are passed as SQL parameters.
- SQL safety checks reject write operations and stacked statements.
- Retrieval evidence cannot directly produce SQL.
- Nested boolean filter expressions support AND/OR/NOT and operators such as
  IN, LIKE, and IS NULL.
- Typed subqueries support IN/NOT IN/EXISTS, and CTEs preserve parameterized
  compilation and nested validation.
- Window expressions support whitelisted ranking and analytic functions.
- Joins are represented in the AST and validated before compilation.
- Benchmark cases can compare ordered or unordered execution results with a
  configurable floating-point tolerance.
- SQLite execution is read-only and guarded by a row limit and timeout.
- MCP-style tools call the same engine methods and cannot bypass validation.

## Development

Useful commands:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

Benchmark utilities are documented in `BENCHMARKS.md`.

## Key Modules

```text
src/nlp_sql/engine.py                  orchestration and safe pipeline
src/nlp_sql/parser.py                  deterministic semantic parser
src/nlp_sql/query_ast.py               typed semantic query model
src/nlp_sql/retrieval/                 hybrid retrieval, vector index, store, telemetry
src/nlp_sql/dataset_registry.py        multi-dataset metadata registry
src/nlp_sql/schema_graph.py            transparent schema metadata graph
src/nlp_sql/evaluation/                ambiguity-aware evaluation framework
src/nlp_sql/benchmarks.py              workload and ablation utilities
src/nlp_sql/mcp_interface.py           optional MCP-style facade
```

## Documentation

- `ARCHITECTURE.md`: original deterministic design
- `ARCHITECTURE_V2.md`: hybrid semantic architecture
- `SYSTEM_DESIGN.md`: implementation overview
- `PAPER_MAPPING.md`: research inspiration and non-reproduction boundaries
- `BENCHMARKS.md`: workload and ablation guidance
