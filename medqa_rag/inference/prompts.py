from __future__ import annotations

from medqa_rag.config import InferenceConfig, PromptMode
from medqa_rag.data.medqa_loader import QuestionRecord
from medqa_rag.retrieval.base import RetrievalResult


def build_baseline_prompt(question: QuestionRecord, cfg: InferenceConfig) -> str:
    options_block = _format_options(question)
    if cfg.prompt_mode is PromptMode.COT:
        return (
            "You are a medical assistant. Think step by step about the clinical findings, "
            "reason through each option, and then give the final single-letter answer.\n\n"
            f"Question: {question.question}\n"
            f"Options:\n{options_block}\n\n"
            "Step-by-step reasoning:"
        )

    return (
        "You are a medical assistant. Choose the single best option. "
        "Output ONLY the single letter (A/B/C/D).\n\n"
        f"Question: {question.question}\n"
        f"Options:\n{options_block}\n\n"
        "Answer (letter only):"
    )


def build_prompt(
    question: QuestionRecord,
    passages: list[RetrievalResult],
    cfg: InferenceConfig,
) -> str:
    options_block = _format_options(question)
    evidence_block = _format_evidence(passages[: cfg.top_k_passages], cfg.passage_max_tokens)

    if cfg.prompt_mode is PromptMode.COT:
        return (
            "You are a medical assistant. Use the evidence below to answer the question. "
            "Think step by step: identify the key clinical findings, reason through each option, "
            "and then choose the single best answer.\n\n"
            f"Question: {question.question}\n"
            f"Options:\n{options_block}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Step-by-step reasoning:"
        )

    return (
        "You are a medical assistant. Use the evidence below to choose the single best option. "
        "Output ONLY the single letter (A/B/C/D).\n\n"
        f"Question: {question.question}\n"
        f"Options:\n{options_block}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Answer (letter only):"
    )


def _format_options(question: QuestionRecord) -> str:
    return "\n".join(f"{label}. {text}" for label, text in sorted(question.options.items()))


def _format_evidence(passages: list[RetrievalResult], passage_limit: int) -> str:
    if not passages:
        return "[1] No evidence retrieved."
    lines = []
    for index, passage in enumerate(passages, start=1):
        snippet = truncate_text(passage.text, passage_limit)
        source = passage.source or "unknown"
        lines.append(f"[{index}] {snippet} (Source: {source})")
    return "\n".join(lines)


def truncate_text(text: str, word_limit: int) -> str:
    words = text.split()
    if len(words) <= word_limit:
        return text.strip()
    return " ".join(words[:word_limit]).strip() + " ..."
