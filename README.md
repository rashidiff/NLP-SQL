# NLP-SQL

Deterministic, rule-based natural-language to SQL engine.

This project intentionally does not use LLMs, embeddings, vector databases, or
external AI services. English business queries are normalized and parsed into a
typed Query AST, validated, and then compiled into parameterized SQL.

