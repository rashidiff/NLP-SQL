# Architecture V2

NLP-SQL remains a deterministic `QueryAST`-centered system. The V2 architecture
adds semantic retrieval, adaptive vector telemetry, disk-backed embedding
storage, multi-dataset metadata, and ambiguity-aware evaluation around that
contract.

```text
                     Natural Language
                           |
                           v
                  Dataset Retrieval
                    /           \
               Lexical        Embedding
                    \           /
                     Hybrid Ranker
                           |
                           v
                   Schema Graph Expansion
                           |
                           v
                 Deterministic Parser
                           |
                           v
                      QueryAST
                           |
                     Validator
                           |
                       Compiler
                           |
                  Safe Parameterized SQL
                           |
                       Executor
                           |
                         Result
```

Telemetry from retrieval and execution can feed the adaptive policy, but it must
never bypass `QueryAST` validation or SQL safety validation.

## Retrieval Layer

`src/nlp_sql/retrieval/` contains:

- `base.py`: `SchemaDocument`, `RetrievalCandidate`, and retriever contracts.
- `lexical.py`: deterministic alias/token overlap retrieval.
- `embeddings.py`: local embedding backends and SQLite-backed embedding store.
- `vector_index.py`: replaceable vector index interface and exact baseline.
- `adaptive.py`: interpretable budget policy and workload-shift detection.
- `hybrid.py`: hybrid lexical/embedding ranking.
- `telemetry.py`: retrieval event statistics.

The default backend is local and deterministic. `sentence-transformers` is an
optional local backend, not an external API.

## Multi-Dataset Path

`DatasetRegistry` describes datasets with identifiers, titles, descriptions,
sources, SQLite database paths, schemas, and semantic metadata. The smart-city
demo includes property records, traffic incidents, and public facilities.

The current system does not claim arbitrary cross-dataset join planning. The
metadata model is designed so future planners can reason over multiple sources.

## Schema Graph

`SchemaGraph` represents datasets, tables, columns, and concepts as nodes.
Edges include:

- `belongs_to`
- `foreign_key`
- `semantic_alias`
- `similar_to`
- `references`

Traversal expands retrieved schema candidates transparently. There is no GNN or
opaque learned graph model.

## Adaptive Interpretation

`AdaptiveDecision` exposes candidate interpretations, evidence, selected
candidate, confidence, and fallback reason. The current implementation ranks the
deterministic parser output with retrieval and execution evidence. It does not
silently change semantics after execution.
