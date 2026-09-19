# No-AI Policy

This repository is intentionally deterministic and rule-based.

It must not depend on or import:

- LLM APIs
- generative AI SDKs
- embedding models
- vector databases
- machine-learning frameworks for NLP-to-SQL parsing
- hosted AI services

The allowed pipeline is:

```text
English text
→ normalization
→ tokenization / phrase matching
→ schema-aware semantic extraction
→ typed Query AST
→ validation
→ parameterized SQL
→ optional SQLite execution
```

The HTTP API is not an AI API. It is only a JSON interface around the same
deterministic parser and SQLite execution path exposed by the CLI.

Current runtime dependency:

- `kagglehub`: downloads the public Kaggle CSV dataset.

Tests enforce this policy by scanning project dependencies and source imports
for known AI/ML packages.

