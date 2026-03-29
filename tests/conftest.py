from __future__ import annotations

import pytest

from medqa_rag.config import ChunkConfig, PipelineConfig
from medqa_rag.data.medqa_loader import QuestionRecord
from medqa_rag.data.preprocess import Document, chunk_documents
from medqa_rag.inference.llm_client import LLMResponse


class FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class RuleBasedLLMClient:
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        del temperature
        lowered = prompt.lower()
        if "first trimester" in lowered and "propylthiouracil" in lowered:
            return LLMResponse(text="Answer: C", raw={"mode": "rule"})
        if "graves disease" in lowered and "methimazole" in lowered:
            return LLMResponse(text="B", raw={"mode": "rule"})
        return LLMResponse(text="A", raw={"mode": "rule"})


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str):
            del model_name
            return FakeTokenizer()

    monkeypatch.setattr("medqa_rag.data.preprocess.AutoTokenizer", FakeAutoTokenizer)


@pytest.fixture()
def sample_documents() -> list[Document]:
    return [
        Document(
            id="doc-graves",
            source="statpearls",
            title="Graves Disease",
            text=(
                "Methimazole is first-line medical therapy for Graves disease in nonpregnant adults. "
                "Radioiodine can also be considered in select cases."
            ),
        ),
        Document(
            id="doc-pregnancy",
            source="statpearls",
            title="Hyperthyroidism in Pregnancy",
            text=(
                "Propylthiouracil is preferred during the first trimester of pregnancy for hyperthyroidism. "
                "Methimazole is usually avoided early in pregnancy."
            ),
        ),
        Document(
            id="doc-noise",
            source="wikipedia",
            title="Levothyroxine",
            text="Levothyroxine is used to treat hypothyroidism and replace thyroid hormone.",
        ),
    ]


@pytest.fixture()
def sample_chunks(sample_documents) -> list[dict]:
    return chunk_documents(sample_documents, ChunkConfig(max_tokens=18, overlap=4, min_chunk_chars=10), model_name="fake-model")


@pytest.fixture()
def sample_questions() -> list[QuestionRecord]:
    return [
        QuestionRecord(
            id="q1",
            question="A 28-year-old nonpregnant woman with Graves disease needs initial medical therapy. Which drug is preferred?",
            options={
                "A": "Radioiodine",
                "B": "Methimazole",
                "C": "Propylthiouracil",
                "D": "Levothyroxine",
            },
            answer="B",
        ),
        QuestionRecord(
            id="q2",
            question="A pregnant patient in the first trimester has hyperthyroidism. Which drug is preferred?",
            options={
                "A": "Levothyroxine",
                "B": "Methimazole",
                "C": "Propylthiouracil",
                "D": "Radioiodine",
            },
            answer="C",
        ),
    ]


@pytest.fixture()
def default_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.chunk = ChunkConfig(max_tokens=18, overlap=4, min_chunk_chars=10)
    cfg.retrieval.dense_k = 3
    cfg.retrieval.bm25_k = 3
    cfg.rerank.top_k = 3
    cfg.inference.top_k_passages = 2
    return cfg


@pytest.fixture()
def rule_based_llm() -> RuleBasedLLMClient:
    return RuleBasedLLMClient()
