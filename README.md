# NLP-SQL

NLP-SQL is a deterministic, safety-oriented natural-language query engine for
turning supported English questions into validated, parameterized SQL.

The central contract is still `QueryAST`. Natural language never becomes SQL
directly. Every query must pass through structured parsing, AST validation,
parameterized compilation, and SQL safety validation before execution.

The project has evolved into a hybrid semantic query system:

- deterministic rule-based parsing for supported grammar;
- schema retrieval over datasets, tables, columns, aliases, descriptions, and
  representative values;
- local embedding retrieval for schema matching only, never SQL generation;
- adaptive vector search telemetry and interpretable search-budget policy;
- SQLite-backed embedding storage with a bounded in-memory LRU cache;
- ambiguity-aware Text-to-SQL evaluation utilities;
- multi-dataset registry and lightweight schema graph expansion;
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
  -> deterministic semantic parser
  -> typed QueryAST
  -> AST validation
  -> parameterized SQL compilation
  -> SQL safety validation
  -> optional SQLite execution
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

## CLI Usage

Prepare the Nashville Housing SQLite database:

```bash
nlp-sql prepare-data
```

Parse a query and show generated AST/SQL:

```bash
nlp-sql parse "show properties with bedrooms at least 4"
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
- User values are passed as SQL parameters.
- SQL safety checks reject write operations and stacked statements.
- Retrieval evidence cannot directly produce SQL.
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
