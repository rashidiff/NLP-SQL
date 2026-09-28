from pathlib import Path

from nlp_sql.retrieval import HybridSchemaRetriever, documents_from_schema
from nlp_sql.retrieval.adaptive import AdaptiveVectorIndex
from nlp_sql.retrieval.embeddings import EmbeddingStore, LocalHashEmbeddingBackend
from nlp_sql.retrieval.vector_index import InMemoryVectorIndex
from nlp_sql.schema import SchemaRegistry


def test_hybrid_retrieval_returns_schema_candidates() -> None:
    documents = documents_from_schema(SchemaRegistry.default())
    retriever = HybridSchemaRetriever(documents)

    candidates = retriever.search("total revenue by customer")

    assert candidates
    assert any(candidate.document.column == "amount" for candidate in candidates)


def test_embedding_store_reuses_persisted_vectors(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "embeddings.sqlite", max_cache_items=1)
    backend = LocalHashEmbeddingBackend(dimensions=16)

    records = store.get_or_embed(
        {"orders/amount": "sales revenue amount"},
        backend=backend,
        source_hashes={"orders/amount": "hash-1"},
    )
    cached = store.get_or_embed(
        {"orders/amount": "sales revenue amount"},
        backend=backend,
        source_hashes={"orders/amount": "hash-1"},
    )

    assert records["orders/amount"].vector == cached["orders/amount"].vector


def test_adaptive_vector_index_records_stats() -> None:
    backend = LocalHashEmbeddingBackend(dimensions=16)
    index = AdaptiveVectorIndex(InMemoryVectorIndex())
    vectors = backend.embed(("sale price", "traffic incident"))
    index.insert("housing/sale_price", vectors[0])
    index.insert("traffic/incident", vectors[1])

    results = index.search("expensive houses", vectors[0], top_k=1)

    assert results[0].key == "housing/sale_price"
    assert index.stats()["query_count"] == 1
