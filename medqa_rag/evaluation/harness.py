from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Callable

from medqa_rag.config import PipelineConfig
from medqa_rag.data.medqa_loader import QuestionRecord
from medqa_rag.inference.parser import parse_answer_letter
from medqa_rag.inference.prompts import build_baseline_prompt, build_prompt
from medqa_rag.retrieval.base import BaseRetriever
from medqa_rag.retrieval.query import build_query


def _process_baseline_question(
    question: QuestionRecord,
    llm_client,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    prompt = build_baseline_prompt(question, cfg.inference)
    response = llm_client.generate(prompt, temperature=cfg.inference.temperature)
    prediction = parse_answer_letter(response.text)
    return {
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


def _process_rag_question(
    question: QuestionRecord,
    retriever: BaseRetriever,
    llm_client,
    cfg: PipelineConfig,
    reranker=None,
) -> dict[str, Any]:
    retrieval_query = build_query(question, mode=cfg.retrieval.query_formulation)
    candidate_k = max(cfg.retrieval.dense_k, cfg.retrieval.bm25_k, cfg.rerank.top_k, cfg.inference.top_k_passages)
    candidates = retriever.retrieve(retrieval_query, candidate_k)
    if reranker is not None and cfg.rerank.enabled:
        candidates = reranker.rerank(retrieval_query, candidates, cfg.rerank.top_k)
    selected = candidates[: cfg.inference.top_k_passages]
    prompt = build_prompt(question, selected, cfg.inference)
    response = llm_client.generate(prompt, temperature=cfg.inference.temperature)
    prediction = parse_answer_letter(response.text)
    return {
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


def evaluate_baseline(
    questions: Iterable[QuestionRecord],
    llm_client,
    cfg: PipelineConfig,
) -> list[dict]:
    question_list = list(questions)
    max_workers = cfg.inference.max_workers

    if max_workers <= 1:
        rows = []
        for question in question_list:
            rows.append(_process_baseline_question(question, llm_client, cfg))
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_baseline_question, q, llm_client, cfg): q
            for q in question_list
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda x: x["id"])
    return rows


def evaluate_rag(
    questions: Iterable[QuestionRecord],
    retriever: BaseRetriever,
    llm_client,
    cfg: PipelineConfig,
    reranker=None,
) -> list[dict]:
    question_list = list(questions)
    max_workers = cfg.inference.max_workers

    if max_workers <= 1:
        rows = []
        for question in question_list:
            rows.append(_process_rag_question(question, retriever, llm_client, cfg, reranker))
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_rag_question, q, retriever, llm_client, cfg, reranker): q
            for q in question_list
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda x: x["id"])
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
