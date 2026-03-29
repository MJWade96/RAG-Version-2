"""
一键运行完整 RAG 流程脚本

使用方式:
    python -m scripts.run_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medqa_rag import load_config
from medqa_rag.data.preprocess import chunk_documents, load_documents, save_chunks
from medqa_rag.data.statpearls_dataset import build_statpearls_dataset
from medqa_rag.retrieval.embedder import Embedder
from medqa_rag.retrieval.faiss_index import save_dense_index


CONFIG_PATH = "configs/default.yaml"
STATPEARLS_DIR = "data/statpearls"
STATPEARLS_COMBINED = f"{STATPEARLS_DIR}/statpearls_combined.json"
DOCUMENTS_PATH = "data/raw/documents.jsonl"
CHUNKS_PATH = "data/processed/chunks.jsonl"
INDEX_DIR = "data/index"


def convert_statpearls_to_documents() -> int:
    """将 StatPearls JSON 转换为 documents.jsonl 格式"""
    import json

    with open(STATPEARLS_COMBINED, "r", encoding="utf-8") as f:
        data = json.load(f)

    Path(DOCUMENTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        for item in data:
            text = item.get("contents") or item.get("content", "")
            doc = {
                "id": item["id"],
                "text": text,
                "title": item.get("title", ""),
                "source": "statpearls",
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    return count


def run_pipeline(
    skip_download: bool = False,
    skip_preprocess: bool = False,
    skip_embed: bool = False,
) -> None:
    """运行完整流程"""
    print("=" * 60)
    print("MedQA RAG 一键运行脚本")
    print("=" * 60)

    # Step 1: 下载 StatPearls
    if not skip_download:
        print("\n[1/4] 下载 StatPearls 数据集...")
        result = build_statpearls_dataset(Path(STATPEARLS_DIR))
        print(f"      下载完成: {result['article_count']} 篇文章, {result['chunk_count']} 个 chunks")

        print("\n[2/4] 转换格式...")
        count = convert_statpearls_to_documents()
        print(f"      转换完成: {count} 个文档")
    else:
        print("\n[1/4] 跳过下载 (--skip-download)")
        print("\n[2/4] 跳过转换格式 (--skip-download)")

    # Step 3: 预处理
    if not skip_preprocess:
        print("\n[3/4] 预处理文档 (生成 chunks)...")
        cfg = load_config(CONFIG_PATH)
        documents = load_documents(DOCUMENTS_PATH)
        chunks = chunk_documents(documents, cfg.chunk, model_name=cfg.retrieval.embedding_model)
        save_chunks(chunks, CHUNKS_PATH)
        print(f"      生成 {len(chunks)} 个 chunks -> {CHUNKS_PATH}")
    else:
        print("\n[3/4] 跳过预处理 (--skip-preprocess)")

    # Step 4: 生成向量索引
    if not skip_embed:
        print("\n[4/4] 生成向量索引...")
        cfg = load_config(CONFIG_PATH)
        from medqa_rag.data.preprocess import load_chunks

        chunks = load_chunks(CHUNKS_PATH)
        texts = [chunk["chunk_text"] for chunk in chunks]
        embedder = Embedder(cfg.retrieval.embedding_model, device=cfg.retrieval.embedding_device)
        vectors = embedder.encode(texts)
        save_dense_index(INDEX_DIR, vectors, chunks)
        print(f"      索引保存完成: {len(chunks)} 个向量 -> {INDEX_DIR}")
    else:
        print("\n[4/4] 跳过向量嵌入 (--skip-embed)")

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)
    print("\n下一步运行评估:")
    print("  基线评估: python -m scripts.run_baseline")
    print("  RAG 评估: python -m scripts.run_rag")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MedQA RAG 一键运行脚本")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载步骤 (已有数据)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过预处理步骤",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="跳过向量嵌入步骤",
    )

    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess,
        skip_embed=args.skip_embed,
    )


if __name__ == "__main__":
    main()