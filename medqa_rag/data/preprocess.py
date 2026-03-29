from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from transformers import AutoTokenizer

from medqa_rag.config import ChunkConfig


@dataclass(slots=True)
class Document:
    id: str
    text: str
    source: str = "unknown"
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "metadata": self.metadata,
        }


class HFTokenizerAdapter:
    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("model_name must be provided for chunking.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(list(tokens))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_chunker(cfg: ChunkConfig, model_name: str | None = None) -> Callable[[str], list[str]]:
    tokenizer = _load_tokenizer(model_name)

    def chunk_text(text: str) -> list[str]:
        return [chunk["chunk_text"] for chunk in _chunk_text_with_metadata(text, cfg, tokenizer)]

    return chunk_text


def chunk_documents(
    documents: Iterable[Document],
    cfg: ChunkConfig,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    tokenizer = _load_tokenizer(model_name)
    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunk_payloads = _chunk_text_with_metadata(document.text, cfg, tokenizer)
        for index, chunk in enumerate(chunk_payloads):
            chunks.append(
                {
                    "id": f"{document.id}:{index}",
                    "chunk_text": chunk["chunk_text"],
                    "doc_id": document.id,
                    "source": document.source,
                    "title": document.title,
                    "chunk_id": str(index),
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "metadata": document.metadata,
                }
            )
    return chunks


def load_documents(path: str | Path) -> list[Document]:
    source = Path(path)
    records: list[dict[str, Any]]
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload if isinstance(payload, list) else payload.get("documents", [])
    return [
        Document(
            id=str(record.get("id") or record.get("doc_id") or ""),
            text=str(record.get("text") or record.get("content") or ""),
            source=str(record.get("source") or "unknown"),
            title=str(record.get("title") or ""),
            metadata=dict(record.get("metadata") or {}),
        )
        for record in records
    ]


def save_chunks(chunks: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_tokenizer(model_name: str | None) -> HFTokenizerAdapter:
    return HFTokenizerAdapter(model_name or "")


def _chunk_text_with_metadata(
    text: str,
    cfg: ChunkConfig,
    tokenizer: HFTokenizerAdapter,
) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = tokenizer.tokenize(cleaned)
    if not tokens:
        return []

    step = max(1, cfg.max_tokens - cfg.overlap)
    chunks: list[dict[str, Any]] = []
    search_start = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + cfg.max_tokens]
        chunk_text = clean_text(tokenizer.detokenize(window))
        if len(chunk_text) < cfg.min_chunk_chars:
            continue
        char_start, char_end = _locate_chunk(cleaned, chunk_text, search_start)
        search_start = max(search_start, char_start + 1)
        chunks.append(
            {
                "chunk_text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
            }
        )
    return chunks


def _locate_chunk(text: str, chunk_text: str, hint: int) -> tuple[int, int]:
    probe = chunk_text[: min(len(chunk_text), 32)]
    if not probe:
        return hint, hint
    index = text.find(probe, hint)
    if index == -1:
        index = text.find(probe)
    if index == -1:
        index = hint
    return index, min(len(text), index + len(chunk_text))
