from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(slots=True)
class QuestionRecord:
    id: str
    question: str
    options: dict[str, str]
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["answer"] = self.answer.upper() if self.answer else None
        payload["options"] = {key.upper(): value for key, value in self.options.items()}
        return payload


def load_medqa(path: str | Path) -> list[QuestionRecord]:
    records = _load_serialized(path)
    return [normalize_question_record(record) for record in records]


def split_records(
    records: list[QuestionRecord],
    dev_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    import random

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    boundary = int(len(shuffled) * dev_ratio)
    return shuffled[:boundary], shuffled[boundary:]


def write_medqa(records: Iterable[QuestionRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def normalize_question_record(payload: dict[str, Any]) -> QuestionRecord:
    options = payload.get("options")
    if not options:
        options = {
            label: payload[label]
            for label in OPTION_LABELS[:6]
            if payload.get(label) is not None
        }
    normalized_options = normalize_options(options)
    answer = payload.get("answer") or payload.get("label") or payload.get("gold")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "question", "options", "answer", "label", "gold", *OPTION_LABELS}
    }
    return QuestionRecord(
        id=str(payload.get("id") or payload.get("qid") or payload.get("question_id") or ""),
        question=str(payload.get("question") or payload.get("stem") or "").strip(),
        options=normalized_options,
        answer=str(answer).strip().upper() if answer else None,
        metadata=metadata,
    )


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        pairs = options.items()
    elif isinstance(options, list):
        pairs = []
        for index, value in enumerate(options):
            label = OPTION_LABELS[index]
            if isinstance(value, dict):
                label = str(value.get("label") or value.get("key") or label).upper()
                text = str(value.get("text") or value.get("value") or "").strip()
            else:
                text = str(value).strip()
            pairs.append((label, text))
    else:
        raise TypeError(f"Unsupported options payload: {type(options)!r}")

    normalized: dict[str, str] = {}
    for key, value in pairs:
        key_text = str(key).strip().upper()
        if not key_text:
            continue
        normalized[key_text[0]] = str(value).strip()
    return dict(sorted(normalized.items()))


def _load_serialized(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported MedQA file structure in {source}")
