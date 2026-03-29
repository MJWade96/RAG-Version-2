from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    source: str
    title: str = ""
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return top-k retrieval results for the given query."""
