from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult
from medqa_rag.retrieval.embedder import Embedder, save_embeddings


class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        faiss_index: faiss.Index,
        metadata: list[dict],
        embedder: Embedder,
    ) -> None:
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedder = embedder

    @classmethod
    def from_chunks(cls, chunks: list[dict], embedder: Embedder) -> "FaissRetriever":
        texts = [chunk.get("chunk_text") or chunk.get("text") or "" for chunk in chunks]
        vectors = _normalize_rows(embedder.encode(texts))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(faiss_index=index, metadata=chunks, embedder=embedder)

    @classmethod
    def from_index_dir(cls, index_dir: str | Path, embedder: Embedder) -> "FaissRetriever":
        root = Path(index_dir)
        index_path = root / "dense.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        metadata = _load_metadata(root / "dense_meta.jsonl")
        return cls(faiss_index=faiss.read_index(str(index_path)), metadata=metadata, embedder=embedder)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.metadata:
            return []

        query_vector = _normalize_rows(self.embedder.encode([query]))
        scores, indices = self.faiss_index.search(query_vector, min(top_k, len(self.metadata)))

        results = []
        for index, score in zip(indices[0].tolist(), scores[0].tolist()):
            if index < 0:
                continue
            meta = self.metadata[index]
            results.append(
                RetrievalResult(
                    chunk_id=str(meta.get("id") or meta.get("chunk_id") or index),
                    score=float(score),
                    text=str(meta.get("chunk_text") or meta.get("text") or ""),
                    source=str(meta.get("source") or "unknown"),
                    title=str(meta.get("title") or ""),
                    doc_id=str(meta.get("doc_id") or ""),
                    metadata=dict(meta.get("metadata") or {}),
                )
            )
        return results


def save_dense_index(index_dir: str | Path, vectors: np.ndarray, metadata: Iterable[dict]) -> Path:
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_rows(vectors)
    save_embeddings(normalized, root / "dense_vectors.npy")
    meta_path = root / "dense_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as handle:
        for record in metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    faiss.write_index(index, str(root / "dense.faiss"))
    return root


def _load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms
