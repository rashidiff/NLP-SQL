# Benchmarks

The benchmark utilities are intentionally lightweight and reproducible. They
support workload-shift experiments and ablations without changing the safe query
pipeline.

## Workload Generator

```python
from pathlib import Path

from nlp_sql.benchmarks import BenchmarkRunner, WorkloadGenerator
from nlp_sql.engine import NlpSqlEngine

engine = NlpSqlEngine.for_housing_dataset()
generator = WorkloadGenerator(seed=17)
runner = BenchmarkRunner(engine)

uniform = generator.uniform(100)
skewed = generator.skewed(100, hot_topic="housing", hot_share=0.8)
shift = generator.sudden_shift(100, before="housing", after="traffic")

reports = (
    runner.run("uniform", uniform),
    runner.run("skewed", skewed),
    runner.run("sudden-shift", shift),
)
runner.write_reports(reports, Path("reports"))
```

By default the runner measures parsing/planning only. To include guarded SQLite
execution, row-limit handling, and execution-guided repair in the measurements:

```python
execution_runner = BenchmarkRunner(engine, execute_queries=True)
report = execution_runner.run("uniform-execution", uniform)
```

Reported fields:

- query count
- p50 latency
- p95 latency
- success rate
- retrieval telemetry snapshot
- LLM, repair, clarification, and execution-failure counts
- executed query count (when execution mode is enabled)

## Ablation Runner

```python
from nlp_sql.benchmarks import WorkloadGenerator, run_ablation_suite
from nlp_sql.engine import NlpSqlEngine

engine = NlpSqlEngine.for_housing_dataset()
workload = WorkloadGenerator(seed=17).uniform(50)
results = run_ablation_suite(engine, workload)
```

The helper currently provides named comparison reports over the same configured
engine. The labels are report names, not runtime feature switches; a true
component ablation should construct separate engines with the desired planner
and retriever configuration.

Current report labels:

- deterministic rules only
- rules plus hybrid retrieval evidence
- hybrid adaptive retrieval

Future experiments can add lexical-only retrieval, fixed semantic retrieval,
schema graph on/off, and cache on/off variants.

## Dynamic Workload Scenarios

Supported workload shapes:

- uniform
- skewed
- sudden topic shift
- gradual topic shift

The adaptive vector layer records margins and hit distributions so workload
shift detection can be inspected rather than inferred from a black-box model.
