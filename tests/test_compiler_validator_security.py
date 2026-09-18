from nlp_sql.compiler import SqlCompiler
from nlp_sql.query_ast import (
    AggregationExpression,
    ColumnExpression,
    HavingPredicate,
    QueryAST,
    QueryType,
)
from nlp_sql.safety import SafetyValidator
from nlp_sql.schema import SchemaRegistry
from nlp_sql.validator import ASTValidator


def test_compiler_parameterizes_having_values() -> None:
    ast = QueryAST(
        type=QueryType.SELECT,
        source="orders",
        select=(ColumnExpression("column", "customer_id"),),
        aggregations=(AggregationExpression("aggregation", "SUM", "amount", "total_purchase"),),
        having=(
            HavingPredicate(
                AggregationExpression("aggregation", "SUM", "amount"),
                ">",
                5_000_000,
            ),
        ),
        group_by=("customer_id",),
    )

    error = ASTValidator(SchemaRegistry.default()).validate(ast)
    compiled = SqlCompiler().compile(ast)

    assert error is None
    assert compiled.sql == (
        "SELECT customer_id, SUM(amount) AS total_purchase FROM orders "
        "GROUP BY customer_id HAVING SUM(amount) > ?"
    )
    assert compiled.parameters == (5_000_000,)


def test_validator_rejects_unknown_columns() -> None:
    ast = QueryAST(
        type=QueryType.SELECT,
        source="orders",
        select=(ColumnExpression("column", "password"),),
    )

    error = ASTValidator(SchemaRegistry.default()).validate(ast)

    assert error is not None
    assert error.code == "UNKNOWN_COLUMN"


def test_safety_validator_rejects_write_keywords() -> None:
    error = SafetyValidator().validate_input("show customers delete from customers")

    assert error is not None
    assert error.code == "FORBIDDEN_OPERATION"
