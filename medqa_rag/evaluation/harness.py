from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from medqa_rag.config import PipelineConfig
from medqa_rag.data.medqa_loader import QuestionRecord
from medqa_rag.inference.parser import parse_answer_letter
from medqa_rag.inference.prompts import build_baseline_prompt, build_prompt
from medqa_rag.retrieval.base import BaseRetriever
from medqa_rag.retrieval.query import build_query


def evaluate_baseline(
    questions: Iterable[QuestionRecord],
    llm_client,
    cfg: PipelineConfig,
) -> list[dict]:
    rows = []
    for question in questions:
        prompt = build_baseline_prompt(question, cfg.inference)
        response = llm_client.generate(prompt, temperature=cfg.inference.temperature)
        prediction = parse_answer_letter(response.text)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "options": question.options,
                "answer": question.answer,
                "prediction": prediction,
                "baseline_pred": prediction,
                "prompt": prompt,
                "response_text": response.text,
                "correct": prediction == question.answer if question.answer else None,
            }
        )
    return rows


def evaluate_rag(
    questions: Iterable[QuestionRecord],
    retriever: BaseRetriever,
    llm_client,
    cfg: PipelineConfig,
    reranker=None,
) -> list[dict]:
    rows = []
    candidate_k = max(cfg.retrieval.dense_k, cfg.retrieval.bm25_k, cfg.rerank.top_k, cfg.inference.top_k_passages)
    for question in questions:
        retrieval_query = build_query(question, mode=cfg.retrieval.query_formulation)
        candidates = retriever.retrieve(retrieval_query, candidate_k)
        if reranker is not None and cfg.rerank.enabled:
            candidates = reranker.rerank(retrieval_query, candidates, cfg.rerank.top_k)
        selected = candidates[: cfg.inference.top_k_passages]
        prompt = build_prompt(question, selected, cfg.inference)
        response = llm_client.generate(prompt, temperature=cfg.inference.temperature)
        prediction = parse_answer_letter(response.text)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "options": question.options,
                "answer": question.answer,
                "prediction": prediction,
                "rag_pred": prediction,
                "retrieval_query": retrieval_query,
                "retrieved_ids": [item.chunk_id for item in selected],
                "retrieved_scores": [item.score for item in selected],
                "retrieved_sources": [item.source for item in selected],
                "prompt": prompt,
                "response_text": response.text,
                "correct": prediction == question.answer if question.answer else None,
            }
        )
    return rows


def write_results(rows: Iterable[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
