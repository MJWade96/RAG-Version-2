from medqa_rag.config import ChunkConfig
from medqa_rag.data.preprocess import chunk_documents


def test_chunk_documents_preserves_overlap(sample_documents):
    chunks = chunk_documents([sample_documents[0]], ChunkConfig(max_tokens=8, overlap=2, min_chunk_chars=5), model_name="fake-model")
    assert len(chunks) >= 2
    first_tokens = chunks[0]["chunk_text"].split()
    second_tokens = chunks[1]["chunk_text"].split()
    assert first_tokens[-2:] == second_tokens[:2]
    assert chunks[0]["char_start"] < chunks[0]["char_end"]
    assert chunks[0]["doc_id"] == "doc-graves"
