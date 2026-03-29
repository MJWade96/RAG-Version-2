"""Retrieval building blocks."""

from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult
from medqa_rag.retrieval.bm25 import BM25Retriever
from medqa_rag.retrieval.embedder import Embedder
from medqa_rag.retrieval.faiss_index import FaissRetriever, save_dense_index
from medqa_rag.retrieval.hybrid import HybridRetriever
from medqa_rag.retrieval.query import build_query

__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "Embedder",
    "FaissRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "build_query",
    "save_dense_index",
]
