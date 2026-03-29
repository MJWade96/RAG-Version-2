from __future__ import annotations

from medqa_rag import load_config
from medqa_rag.data.preprocess import chunk_documents, load_documents, save_chunks


CONFIG_PATH = "configs/default.yaml"
INPUT_PATH = "data/raw/documents.jsonl"
OUTPUT_PATH = "data/processed/chunks.jsonl"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    documents = load_documents(INPUT_PATH)
    chunks = chunk_documents(documents, cfg.chunk, model_name=cfg.retrieval.embedding_model)
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
