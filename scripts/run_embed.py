from __future__ import annotations

from medqa_rag import load_config
from medqa_rag.data.preprocess import load_chunks
from medqa_rag.retrieval.embedder import Embedder
from medqa_rag.retrieval.faiss_index import save_dense_index


CONFIG_PATH = "configs/default.yaml"
CHUNKS_PATH = "data/processed/chunks.jsonl"
OUTPUT_DIR = "data/index"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["chunk_text"] for chunk in chunks]
    embedder = Embedder(cfg.retrieval.embedding_model, device=cfg.retrieval.embedding_device)
    vectors = embedder.encode(texts)
    save_dense_index(OUTPUT_DIR, vectors, chunks)
    print(f"Saved dense index for {len(chunks)} chunks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
