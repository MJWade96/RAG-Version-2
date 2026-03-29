from __future__ import annotations

from dataclasses import replace

from sentence_transformers import CrossEncoder

from medqa_rag.config import RerankConfig
from medqa_rag.rerank.base import BaseReranker
from medqa_rag.retrieval.base import RetrievalResult


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        cfg: RerankConfig,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.model = CrossEncoder(cfg.model, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        scores = self.model.predict([(query, candidate.text) for candidate in candidates], batch_size=self.batch_size)
        reranked = [replace(candidate, score=float(score)) for candidate, score in zip(candidates, scores)]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
