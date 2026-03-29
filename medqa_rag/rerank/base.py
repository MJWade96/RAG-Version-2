from __future__ import annotations

from abc import ABC, abstractmethod

from medqa_rag.retrieval.base import RetrievalResult


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score and re-order candidates."""
