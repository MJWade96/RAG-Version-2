from medqa_rag.config import RerankConfig
from medqa_rag.retrieval.base import RetrievalResult
import medqa_rag.rerank.cross_encoder as cross_encoder_module


def test_cross_encoder_reranker_reorders_candidates(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model: str, device: str = "cpu") -> None:
            del model, device

        def predict(self, pairs, batch_size: int):
            del pairs, batch_size
            return [0.1, 0.9]

    monkeypatch.setattr(cross_encoder_module, "CrossEncoder", FakeCrossEncoder)
    reranker = cross_encoder_module.CrossEncoderReranker(RerankConfig(model="fake-model", batch_size=2))
    candidates = [
        RetrievalResult(chunk_id="1", score=0.9, text="Levothyroxine treats hypothyroidism.", source="wiki"),
        RetrievalResult(chunk_id="2", score=0.2, text="Methimazole is first-line for Graves disease.", source="statpearls"),
    ]
    results = reranker.rerank("Graves disease methimazole", candidates, top_k=2)
    assert results[0].chunk_id == "2"
