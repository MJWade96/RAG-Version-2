from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable

import yaml

from medqa_rag.config import PipelineConfig, PromptMode, QueryFormulation
from medqa_rag.evaluation.harness import evaluate_rag
from medqa_rag.evaluation.stats import accuracy


GRID_OVERRIDES = {
    "embedding_model": ("retrieval", "embedding_model"),
    "query_formulation": ("retrieval", "query_formulation"),
    "fusion_alpha": ("retrieval", "fusion_alpha"),
    "reranker_enabled": ("rerank", "enabled"),
    "prompt_mode": ("inference", "prompt_mode"),
    "top_k_passages": ("inference", "top_k_passages"),
}


def load_experiment_grid(path: str | Path) -> dict[str, list[Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def apply_overrides(cfg: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    payload = deepcopy(cfg.to_dict())
    for key, value in overrides.items():
        section, field_name = GRID_OVERRIDES[key]
        payload[section][field_name] = value

    updated = PipelineConfig.from_dict(payload)
    if isinstance(updated.retrieval.query_formulation, str):
        updated.retrieval.query_formulation = QueryFormulation(updated.retrieval.query_formulation)
    if isinstance(updated.inference.prompt_mode, str):
        updated.inference.prompt_mode = PromptMode(updated.inference.prompt_mode)
    return updated


def run_experiment_grid(
    questions,
    cfg: PipelineConfig,
    grid: dict[str, list[Any]],
    retriever_factory: Callable[[PipelineConfig], Any],
    reranker_factory: Callable[[PipelineConfig], Any],
    llm_client,
) -> list[dict[str, Any]]:
    rows = []
    for combo in iter_grid(grid):
        current_cfg = apply_overrides(cfg, combo)
        retriever = retriever_factory(current_cfg)
        reranker = reranker_factory(current_cfg)
        predictions = evaluate_rag(questions, retriever=retriever, llm_client=llm_client, cfg=current_cfg, reranker=reranker)
        rows.append(
            {
                "config": combo,
                "accuracy": accuracy(predictions),
                "num_questions": len(predictions),
            }
        )
    rows.sort(key=lambda item: item["accuracy"], reverse=True)
    return rows
