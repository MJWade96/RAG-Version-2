"""
Convert StatPearls JSON to documents.jsonl format for RAG preprocessing.

StatPearls output format:
  {"id": "xxx", "title": "...", "content": "...", "contents": "...", "source": "statpearls"}

Required format for preprocess.py:
  {"id": "...", "text": "...", "title": "...", "source": "..."}
"""

import json
from pathlib import Path


def convert_statpearls_to_documents(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Convert StatPearls combined JSON to documents.jsonl format."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        statpearls_data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in statpearls_data:
            # Use 'contents' field which has title concatenated, or fall back to content
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


if __name__ == "__main__":
    INPUT_PATH = "data/statpearls/statpearls_combined.json"
    OUTPUT_PATH = "data/raw/documents.jsonl"

    count = convert_statpearls_to_documents(INPUT_PATH, OUTPUT_PATH)
    print(f"Converted {count} documents to {OUTPUT_PATH}")