from __future__ import annotations

import re
from typing import Mapping

from medqa_rag.config import QueryFormulation
from medqa_rag.data.medqa_loader import QuestionRecord


def build_query(
    question: QuestionRecord | str,
    options: Mapping[str, str] | None = None,
    mode: QueryFormulation = QueryFormulation.QUESTION_ONLY,
) -> str:
    if isinstance(question, QuestionRecord):
        question_text = question.question
        options_map = question.options
    else:
        question_text = question
        options_map = dict(options or {})

    if mode is QueryFormulation.QUESTION_ONLY:
        return normalize_query(question_text)
    if mode is QueryFormulation.QUESTION_PLUS_OPTIONS:
        combined = question_text + " " + " ".join(f"{key} {value}" for key, value in sorted(options_map.items()))
        return normalize_query(combined)
    entities = extract_candidate_entities(question_text + " " + " ".join(options_map.values()))
    return normalize_query(" ".join(entities) or question_text)


def extract_candidate_entities(text: str, limit: int = 8) -> list[str]:
    stopwords = {"with", "from", "that", "this", "patient", "which", "would", "most"}
    candidates = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text):
        lowered = token.lower()
        if lowered in stopwords or lowered in candidates:
            continue
        candidates.append(lowered)
        if len(candidates) >= limit:
            break
    return candidates


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()
