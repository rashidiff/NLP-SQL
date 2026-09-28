from nlp_sql.engine import NlpSqlEngine
from nlp_sql.evaluation import BenchmarkCase, EvaluationRunner
from nlp_sql.query_ast import AggregationExpression, QueryAST, QueryType


def test_evaluation_accepts_one_valid_ambiguous_interpretation() -> None:
    engine = NlpSqlEngine()
    case = BenchmarkCase(
        id="avg-order",
        question="average order amount",
        dataset_id="default",
        valid_asts=(
            QueryAST(
                type=QueryType.SELECT,
                source="orders",
                aggregations=(
                    AggregationExpression("aggregation", "AVG", "amount", "avg_amount"),
                ),
                status="inferred",
            ),
        ),
    )

    report = EvaluationRunner(engine).evaluate((case,))

    assert report.cases[0].success
