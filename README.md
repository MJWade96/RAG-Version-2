# MedQA RAG

`medqa-rag` is a modular retrieval-augmented generation pipeline for MedQA-style multiple-choice evaluation. The repository is structured around the plan in `RAG system plan.md` and is designed to support:

- corpus acquisition and preprocessing
- dense, sparse, and hybrid retrieval
- optional cross-encoder reranking
- baseline and RAG evaluation harnesses
- configuration-driven experiment sweeps
- reproducible result logging and lightweight statistical analysis

## Layout

Core package code lives in `medqa_rag/`. Variable-driven runner scripts live in `scripts/`. Configs live in `configs/`. Tests in `tests/` use mocks for external model clients and tokenizer loading.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Typical Workflow

```bash
python -m scripts.run_preprocess
python -m scripts.run_embed
python -m scripts.run_baseline
python -m scripts.run_rag
```

## Notes

- Before running any script in `scripts/`, edit the variables at the top of the file or set the corresponding environment variables used there.
- The default inference client targets Tianyi Cloud's OpenAI-compatible endpoint at `https://wishub-x6.ctyun.cn/v1` and sends `extra_body={"enable_thinking": True}` by default.
- API calls do not pass `max_tokens`; the service default is used.
