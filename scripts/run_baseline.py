from __future__ import annotations

import os
from pathlib import Path

from medqa_rag import load_config
from medqa_rag.data.medqa_loader import load_medqa
from medqa_rag.evaluation.harness import evaluate_baseline, write_results
from medqa_rag.evaluation.stats import accuracy
from medqa_rag.inference.llm_client import TianyiOpenAILLMClient


CONFIG_PATH = "configs/default.yaml"
MEDQA_PATH = "data/raw/medqa/dev.jsonl"
OUTPUT_PATH = "results/baseline_dev.jsonl"
BASE_URL = os.getenv("XIRANG_base_url", "https://wishub-x6.ctyun.cn/v1")
APP_KEY = os.getenv("XIRANG_app_key", "")
MODEL_ID = os.getenv("XIRANG_model_id", "")
ENABLE_THINKING = True


def main() -> None:
    _require_value("APP_KEY", APP_KEY)
    _require_value("MODEL_ID", MODEL_ID)

    cfg = load_config(CONFIG_PATH)
    cfg.inference.enable_thinking = ENABLE_THINKING
    questions = load_medqa(MEDQA_PATH)
    llm_client = TianyiOpenAILLMClient(
        base_url=BASE_URL,
        api_key=APP_KEY,
        model=MODEL_ID,
        enable_thinking=ENABLE_THINKING,
    )
    rows = evaluate_baseline(questions, llm_client=llm_client, cfg=cfg)
    write_results(rows, OUTPUT_PATH)
    print(f"Baseline accuracy: {accuracy(rows):.4f}")


def _require_value(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be configured in this script or the environment.")


if __name__ == "__main__":
    main()
