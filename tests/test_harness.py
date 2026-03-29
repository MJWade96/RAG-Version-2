from medqa_rag.evaluation.harness import evaluate_baseline, evaluate_rag
from medqa_rag.evaluation.stats import accuracy
from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult


class StaticRetriever(BaseRetriever):
    def __init__(self, mapping: dict[str, list[RetrievalResult]]) -> None:
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        for key, results in self.mapping.items():
            if key in query.lower():
                return results[:top_k]
        return next(iter(self.mapping.values()))[:top_k]


class IdentityReranker:
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        del query
        return candidates[:top_k]


def test_evaluate_baseline_outputs_rows(sample_questions, default_cfg, rule_based_llm):
    rows = evaluate_baseline(sample_questions, llm_client=rule_based_llm, cfg=default_cfg)
    assert len(rows) == 2
    assert rows[0]["baseline_pred"] in {"A", "B", "C", "D"}


def test_evaluate_rag_improves_with_retrieval(sample_questions, default_cfg, rule_based_llm):
    retriever = StaticRetriever(
        {
            "graves disease": [
                RetrievalResult(chunk_id="1", score=0.9, text="Methimazole is first-line for Graves disease.", source="statpearls"),
            ],
            "first trimester": [
                RetrievalResult(
                    chunk_id="2",
                    score=0.95,
                    text="Propylthiouracil is preferred during the first trimester of pregnancy for hyperthyroidism.",
                    source="statpearls",
                ),
            ],
        }
    )
    reranker = IdentityReranker()

    rows = evaluate_rag(
        sample_questions,
        retriever=retriever,
        llm_client=rule_based_llm,
        cfg=default_cfg,
        reranker=reranker,
    )
    assert accuracy(rows) >= 0.5
    assert rows[0]["retrieved_ids"]
