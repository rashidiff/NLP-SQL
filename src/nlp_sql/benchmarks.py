"""Workload and ablation benchmarking utilities."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from nlp_sql.engine import NlpSqlEngine


@dataclass(frozen=True)
class WorkloadQuery:
    text: str
    topic: str


@dataclass(frozen=True)
class WorkloadReport:
    name: str
    query_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    retrieval_stats: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "query_count": self.query_count,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "success_rate": self.success_rate,
            "retrieval_stats": self.retrieval_stats,
        }


class WorkloadGenerator:
    def __init__(self, *, seed: int = 17) -> None:
        self._random = random.Random(seed)
        self._templates = {
            "housing": (
                "show expensive homes",
                "show houses with large lots",
                "average sale price",
                "recent house sales",
            ),
            "traffic": (
                "traffic incidents near schools",
                "most severe traffic incidents",
                "traffic incidents in north area",
            ),
            "facilities": (
                "show schools",
                "public facilities in east area",
                "libraries by area",
            ),
        }

    def uniform(self, count: int) -> tuple[WorkloadQuery, ...]:
        topics = tuple(self._templates)
        return tuple(self._sample(self._random.choice(topics)) for _ in range(count))

    def skewed(self, count: int, *, hot_topic: str = "housing", hot_share: float = 0.8) -> tuple[WorkloadQuery, ...]:
        topics = tuple(self._templates)
        queries: list[WorkloadQuery] = []
        for _ in range(count):
            topic = hot_topic if self._random.random() < hot_share else self._random.choice(topics)
            queries.append(self._sample(topic))
        return tuple(queries)

    def sudden_shift(self, count: int, *, before: str = "housing", after: str = "traffic") -> tuple[WorkloadQuery, ...]:
        midpoint = count // 2
        return tuple(
            self._sample(before if index < midpoint else after) for index in range(count)
        )

    def gradual_shift(self, count: int, *, before: str = "housing", after: str = "traffic") -> tuple[WorkloadQuery, ...]:
        queries: list[WorkloadQuery] = []
        for index in range(count):
            after_probability = index / max(count - 1, 1)
            queries.append(
                self._sample(after if self._random.random() < after_probability else before)
            )
        return tuple(queries)

    def _sample(self, topic: str) -> WorkloadQuery:
        return WorkloadQuery(self._random.choice(self._templates[topic]), topic)


class BenchmarkRunner:
    def __init__(self, engine: NlpSqlEngine) -> None:
        self._engine = engine

    def run(self, name: str, workload: tuple[WorkloadQuery, ...]) -> WorkloadReport:
        latencies: list[float] = []
        successes = 0
        for query in workload:
            started = time.perf_counter()
            result = self._engine.parse(query.text)
            latencies.append((time.perf_counter() - started) * 1000)
            successes += 1 if result.success else 0
        sorted_latencies = sorted(latencies)
        p50 = _percentile(sorted_latencies, 0.50)
        p95 = _percentile(sorted_latencies, 0.95)
        return WorkloadReport(
            name=name,
            query_count=len(workload),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            success_rate=successes / len(workload) if workload else 0.0,
            retrieval_stats=self._engine.retrieval_stats(),
        )

    def write_reports(self, reports: tuple[WorkloadReport, ...], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = [report.to_dict() for report in reports]
        (output_dir / "benchmark-report.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        lines = ["# Benchmark Report", ""]
        for report in reports:
            lines.extend(
                [
                    f"## {report.name}",
                    "",
                    f"- queries: {report.query_count}",
                    f"- p50 latency ms: {report.p50_latency_ms:.3f}",
                    f"- p95 latency ms: {report.p95_latency_ms:.3f}",
                    f"- success rate: {report.success_rate:.3f}",
                    "",
                ]
            )
        (output_dir / "benchmark-report.md").write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class AblationResult:
    name: str
    report: WorkloadReport
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "report": self.report.to_dict(), "notes": self.notes}


def run_ablation_suite(engine: NlpSqlEngine, workload: tuple[WorkloadQuery, ...]) -> tuple[AblationResult, ...]:
    runner = BenchmarkRunner(engine)
    return (
        AblationResult("deterministic_rules_only", runner.run("rules", workload), "Existing parser path."),
        AblationResult("rules_plus_hybrid_retrieval", runner.run("hybrid", workload), "Retrieval evidence attached."),
        AblationResult("hybrid_adaptive_retrieval", runner.run("adaptive", workload), "Adaptive vector budget enabled."),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index]
