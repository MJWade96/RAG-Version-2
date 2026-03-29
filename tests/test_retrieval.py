from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult
from medqa_rag.retrieval.bm25 import BM25Retriever
from medqa_rag.retrieval.hybrid import HybridRetriever


class StaticRetriever(BaseRetriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        del query
        return self.results[:top_k]


def test_bm25_retriever_returns_relevant_chunk(sample_chunks):
    retriever = BM25Retriever(sample_chunks)
    results = retriever.retrieve("first trimester pregnancy hyperthyroidism", top_k=2)
    assert results
    assert "pregnancy" in results[0].text.lower()


def test_hybrid_retriever_combines_dense_and_sparse(default_cfg):
    dense = StaticRetriever(
        [
            RetrievalResult(chunk_id="1", score=0.9, text="Methimazole is first-line for Graves disease.", source="statpearls"),
            RetrievalResult(chunk_id="2", score=0.2, text="Levothyroxine treats hypothyroidism.", source="wiki"),
        ]
    )
    sparse = StaticRetriever(
        [
            RetrievalResult(chunk_id="1", score=1.3, text="Methimazole is first-line for Graves disease.", source="statpearls"),
            RetrievalResult(chunk_id="3", score=0.7, text="Propylthiouracil is used in first trimester pregnancy.", source="statpearls"),
        ]
    )
    hybrid = HybridRetriever(default_cfg.retrieval, dense_retriever=dense, sparse_retriever=sparse)
    results = hybrid.retrieve("Graves disease methimazole", top_k=3)
    assert results
    assert results[0].chunk_id == "1"
    assert any(result.chunk_id == "3" for result in results)
