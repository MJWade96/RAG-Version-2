from __future__ import annotations

from pathlib import Path

from medqa_rag.data.download import extract_medical_entities, fetch_pubmed_abstracts, fetch_wikipedia_pages, save_jsonl


CORPORA = ["wiki", "pubmed"]
QUERIES = []
QUERIES_FILE = ""
OUTPUT_DIR = "data/raw"
NCBI_EMAIL = ""
NCBI_API_KEY = ""


def main() -> None:
    queries = list(QUERIES)
    if QUERIES_FILE:
        queries.extend(
            line.strip()
            for line in Path(QUERIES_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not queries:
        raise ValueError("Set QUERIES or QUERIES_FILE before running this script.")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    if "wiki" in CORPORA:
        entities = []
        for query in queries:
            entities.extend(extract_medical_entities(query))
        pages = fetch_wikipedia_pages(entities)
        save_jsonl(
            [{"id": key, "title": key, "text": value, "source": "wikipedia"} for key, value in pages.items()],
            output_root / "wiki.jsonl",
        )

    if "pubmed" in CORPORA:
        abstracts = fetch_pubmed_abstracts(queries, email=NCBI_EMAIL or None, api_key=NCBI_API_KEY or None)
        save_jsonl(
            [
                {
                    "id": record["pmid"],
                    "title": record["title"],
                    "text": record["abstract"],
                    "source": "pubmed",
                    "metadata": {"query": record["query"]},
                }
                for record in abstracts
            ],
            output_root / "pubmed.jsonl",
        )


if __name__ == "__main__":
    main()
