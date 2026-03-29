from __future__ import annotations

import random
import re
from typing import Iterable

from medqa_rag.data.medqa_loader import QuestionRecord


def compute_evidence_coverage(
    questions: Iterable[QuestionRecord],
    chunks: Iterable[dict],
    sample_size: int | None = 200,
    seed: int = 42,
    min_overlap_terms: int = 2,
) -> dict[str, float | int]:
    question_list = list(questions)
    chunk_list = list(chunks)
    if sample_size is not None and len(question_list) > sample_size:
        question_list = random.Random(seed).sample(question_list, sample_size)

    covered = 0
    for question in question_list:
        if supporting_passages(question, chunk_list, min_overlap_terms=min_overlap_terms):
            covered += 1

    total = len(question_list)
    coverage = covered / total if total else 0.0
    return {"sampled": total, "covered": covered, "coverage": coverage}


def supporting_passages(
    question: QuestionRecord,
    chunks: Iterable[dict],
    min_overlap_terms: int = 2,
    top_k: int = 5,
) -> list[dict]:
    query_terms = _keywords(question.question + " " + " ".join(question.options.values()))
    scored = []
    for chunk in chunks:
        chunk_terms = _keywords(chunk.get("chunk_text", ""))
        overlap = len(query_terms & chunk_terms)
        if overlap >= min_overlap_terms:
            scored.append((overlap, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _keywords(text: str) -> set[str]:
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "patient", "which"}
    return {
        token
        for token in re.findall(r"[a-z0-9-]{3,}", text.lower())
        if token not in stopwords
    }
