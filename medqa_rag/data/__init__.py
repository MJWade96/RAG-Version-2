"""Data loading, download, and preprocessing utilities."""

from medqa_rag.data.coverage_gate import compute_evidence_coverage
from medqa_rag.data.medqa_loader import QuestionRecord, load_medqa
from medqa_rag.data.preprocess import Document, chunk_documents, load_chunks, load_documents, save_chunks

__all__ = [
    "Document",
    "QuestionRecord",
    "chunk_documents",
    "compute_evidence_coverage",
    "load_chunks",
    "load_documents",
    "load_medqa",
    "save_chunks",
]
