from __future__ import annotations

from collections import Counter
import math
import re

from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9-]+")


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_corpus = [self._tokenize(chunk.get("chunk_text") or chunk.get("text") or "") for chunk in chunks]
        self.doc_lens = [len(tokens) for tokens in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_corpus]
        self.doc_freqs = self._compute_doc_freqs()
        self.num_docs = len(self.chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_tokens = self._tokenize(query)
        scores = [self._score_document(query_tokens, index) for index in range(self.num_docs)]
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            RetrievalResult(
                chunk_id=str(self.chunks[index].get("id") or self.chunks[index].get("chunk_id") or index),
                score=float(scores[index]),
                text=str(self.chunks[index].get("chunk_text") or self.chunks[index].get("text") or ""),
                source=str(self.chunks[index].get("source") or "unknown"),
                title=str(self.chunks[index].get("title") or ""),
                doc_id=str(self.chunks[index].get("doc_id") or ""),
                metadata=dict(self.chunks[index].get("metadata") or {}),
            )
            for index in top_indices
        ]

    def _compute_doc_freqs(self) -> Counter:
        document_frequencies: Counter = Counter()
        for tokens in self.tokenized_corpus:
            document_frequencies.update(set(tokens))
        return document_frequencies

    def _score_document(self, query_tokens: list[str], index: int) -> float:
        score = 0.0
        freqs = self.term_freqs[index]
        doc_len = self.doc_lens[index] or 1
        for token in query_tokens:
            if token not in freqs:
                continue
            doc_freq = self.doc_freqs[token]
            idf = math.log(1 + (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            term_freq = freqs[token]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * numerator / denominator
        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())
