from __future__ import annotations

from dataclasses import replace
from statistics import mean, pstdev

from medqa_rag.config import RetrievalConfig, ScoreNormalization
from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        cfg: RetrievalConfig,
        dense_retriever: BaseRetriever | None = None,
        sparse_retriever: BaseRetriever | None = None,
    ) -> None:
        self.cfg = cfg
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense_results = (
            self.dense_retriever.retrieve(query, self.cfg.dense_k)
            if self.dense_retriever and self.cfg.dense_k > 0
            else []
        )
        sparse_results = (
            self.sparse_retriever.retrieve(query, self.cfg.bm25_k)
            if self.sparse_retriever and self.cfg.bm25_k > 0
            else []
        )

        if dense_results and not sparse_results:
            return dense_results[:top_k]
        if sparse_results and not dense_results:
            return sparse_results[:top_k]
        if not dense_results and not sparse_results:
            return []

        dense_norm = _normalize_scores(dense_results, self.cfg.score_normalization)
        sparse_norm = _normalize_scores(sparse_results, self.cfg.score_normalization)
        alpha = self.cfg.fusion_alpha

        merged: dict[str, RetrievalResult] = {}
        fused_scores: dict[str, float] = {}

        for result in dense_results:
            merged[result.chunk_id] = replace(result)
            fused_scores[result.chunk_id] = alpha * dense_norm[result.chunk_id]
        for result in sparse_results:
            merged.setdefault(result.chunk_id, replace(result))
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (1.0 - alpha) * sparse_norm[result.chunk_id]

        ranked = []
        for chunk_id, result in merged.items():
            result.score = fused_scores.get(chunk_id, 0.0)
            ranked.append(result)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]


def _normalize_scores(
    results: list[RetrievalResult],
    method: ScoreNormalization,
) -> dict[str, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    if method is ScoreNormalization.MINMAX:
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return {result.chunk_id: 0.0 for result in results}
        return {result.chunk_id: (result.score - lo) / (hi - lo) for result in results}

    avg = mean(scores)
    std = pstdev(scores)
    if std == 0:
        return {result.chunk_id: 0.0 for result in results}
    return {result.chunk_id: (result.score - avg) / std for result in results}
