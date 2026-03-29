from __future__ import annotations

import json
import os
from pathlib import Path

from medqa_rag import load_config
from medqa_rag.data.medqa_loader import load_medqa
from medqa_rag.data.preprocess import load_chunks
from medqa_rag.experiment import load_experiment_grid, run_experiment_grid
from medqa_rag.inference.llm_client import TianyiOpenAILLMClient
from medqa_rag.rerank.cross_encoder import CrossEncoderReranker
from medqa_rag.retrieval.bm25 import BM25Retriever
from medqa_rag.retrieval.embedder import Embedder
from medqa_rag.retrieval.faiss_index import FaissRetriever
from medqa_rag.retrieval.hybrid import HybridRetriever


CONFIG_PATH = "configs/default.yaml"
GRID_PATH = "configs/experiment_grid.yaml"
MEDQA_PATH = "data/raw/medqa/dev.jsonl"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_PATH = "results/sweep/results.json"
USE_PREBUILT_INDEX = False
INDEX_DIR = "data/index"
BASE_URL = os.getenv("XIRANG_base_url", "https://wishub-x6.ctyun.cn/v1")
APP_KEY = os.getenv("XIRANG_app_key", "")
MODEL_ID = os.getenv("XIRANG_model_id", "")
ENABLE_THINKING = True


def main() -> None:
    _require_value("APP_KEY", APP_KEY)
    _require_value("MODEL_ID", MODEL_ID)

    cfg = load_config(CONFIG_PATH)
    cfg.inference.enable_thinking = ENABLE_THINKING
    grid = load_experiment_grid(GRID_PATH)
    questions = load_medqa(MEDQA_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    llm_client = TianyiOpenAILLMClient(
        base_url=BASE_URL,
        api_key=APP_KEY,
        model=MODEL_ID,
        enable_thinking=ENABLE_THINKING,
        timeout=cfg.inference.timeout,
        rate_limit=cfg.inference.rate_limit,
    )

    def retriever_factory(current_cfg):
        embedder = Embedder(current_cfg.retrieval.embedding_model, device=current_cfg.retrieval.embedding_device)
        if USE_PREBUILT_INDEX:
            dense = FaissRetriever.from_index_dir(INDEX_DIR, embedder=embedder)
        else:
            dense = FaissRetriever.from_chunks(chunks, embedder=embedder)
        sparse = BM25Retriever(chunks)
        return HybridRetriever(current_cfg.retrieval, dense_retriever=dense, sparse_retriever=sparse)

    def reranker_factory(current_cfg):
        if not current_cfg.rerank.enabled:
            return None
        return CrossEncoderReranker(current_cfg.rerank)

    results = run_experiment_grid(
        questions=questions,
        cfg=cfg,
        grid=grid,
        retriever_factory=retriever_factory,
        reranker_factory=reranker_factory,
        llm_client=llm_client,
    )

    target = Path(OUTPUT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} experiment results to {OUTPUT_PATH}")


def _require_value(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be configured in this script or the environment.")


if __name__ == "__main__":
    main()
