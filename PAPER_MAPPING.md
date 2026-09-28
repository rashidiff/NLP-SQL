# Paper Mapping

This project is inspired by several research directions, but it does not claim
to reproduce those systems. The goal is a practical, transparent, deterministic
semantic query engine.

## Fundamental Challenges In Evaluating Text2SQL Solutions And Detecting Their Limitations

Borrowed idea:

- exact SQL-string match is insufficient;
- multiple valid interpretations can exist;
- evaluation should include schema linking, AST structure, execution
  equivalence, ambiguity, failure categories, and slices.

Implemented here:

- `BenchmarkCase` supports multiple valid ASTs and SQL strings;
- schema-linking precision/recall/F1;
- AST structural comparison;
- SQL normalization;
- result-set equivalence helper;
- failure taxonomy and label slices.

Not implemented:

- full benchmark suite from the paper;
- exhaustive natural-language ambiguity annotation;
- large-scale empirical study.

## Quake: Adaptive Indexing For Vector Search

Borrowed idea:

- vector retrieval systems should adapt when access patterns and workloads
  change.

Implemented here:

- `AdaptiveVectorIndex` wraps a vector index;
- tracks query count, latency, hit distribution, margins, selected budget, and
  index size;
- increases search budget for ambiguous low-margin queries;
- detects workload shifts using recent-vs-historical access distribution.

Not implemented:

- Quake's index structure or algorithms;
- distributed ANN;
- formal recall guarantees.

## MLKV: Efficiently Scaling Up Large Embedding Model Training With Disk-Based Key-Value Storage

Borrowed idea:

- embeddings should be stored in a persistent key-value layer instead of being
  recomputed or kept entirely in memory.

Implemented here:

- SQLite-backed `EmbeddingStore`;
- stable keys and source hashes;
- model/version metadata;
- bounded in-memory LRU hot cache;
- batch reads and writes;
- invalidation when source text changes.

Not implemented:

- MLKV's training-time storage architecture;
- GPU training pipeline;
- distributed key-value storage.

## Simple Adaptive Query Processing vs. Learned Query Optimizers

Borrowed idea:

- prefer interpretable runtime evidence and adaptive decisions over opaque
  learned behavior when the system can remain simple.

Implemented here:

- `AdaptiveDecision` exposes candidates, evidence, selected candidate,
  confidence, and fallback reason;
- runtime evidence can include retrieval confidence, parser evidence, execution
  validity, and empty-result observations.

Not implemented:

- PostgreSQL join algorithms from the paper;
- learned query optimizer;
- automatic semantic rewrites after execution.
