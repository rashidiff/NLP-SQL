# Architecture

NLP-SQL uses a deterministic pipeline:

```text
Natural Language
→ Normalized Representation
→ Semantic Parsing
→ Query AST
→ Validation
→ Parameterized SQL
```

The important design decision is that natural language never becomes SQL
directly. The parser produces a semantic `QueryAST`; the validator decides
whether that AST is safe and compatible with the configured schema; the compiler
then converts only validated AST nodes into SQL and parameters.

This separation keeps the system testable and secure:

- Normalization, number parsing, tokenization, dates, schema resolution,
  validation, and compilation are isolated modules.
- SQL generation is independent of NLP parsing.
- The schema registry is the source of truth for allowed tables and columns.
- Safety checks run on user input and generated SQL.
- Future parsers can target the same AST contract.

## Real Database Path

For the Kaggle Nashville Housing dataset, the production path is:

```text
Kaggle CSV
→ SQLite Importer
→ SQLite Schema Introspection
→ SchemaRegistry
→ Rule-Based NLP Parser
→ Query AST
→ AST Validator
→ SQL Compiler
→ SQLiteExecutor
→ Rows
```

The database layer is intentionally optional. `NlpSqlEngine.parse()` and
`NlpSqlEngine.explain()` work without a database connection. `execute()` is the
only method that requires a configured executor, and it still runs only after
AST validation and SQL safety validation.

The intended future shape is:

```text
Rule-Based NLP ───┐
                  ├──→ Query AST → Validator → SQL Compiler
Future Parser ────┘
```

No future parser may bypass AST validation or parameterized SQL compilation.
