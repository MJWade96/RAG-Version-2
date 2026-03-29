from medqa_rag.config import InferenceConfig, PromptMode
from medqa_rag.inference.prompts import build_baseline_prompt, build_prompt
from medqa_rag.retrieval.base import RetrievalResult


def test_direct_prompt_includes_evidence(sample_questions):
    cfg = InferenceConfig(prompt_mode=PromptMode.DIRECT, top_k_passages=2, passage_max_tokens=20)
    passages = [
        RetrievalResult(chunk_id="1", score=1.0, text="Methimazole is first-line therapy.", source="statpearls"),
    ]
    prompt = build_prompt(sample_questions[0], passages, cfg)
    assert "Evidence:" in prompt
    assert "[1]" in prompt
    assert "Answer (letter only):" in prompt


def test_cot_baseline_prompt_requests_reasoning(sample_questions):
    cfg = InferenceConfig(prompt_mode=PromptMode.COT)
    prompt = build_baseline_prompt(sample_questions[0], cfg)
    assert "Step-by-step reasoning:" in prompt
