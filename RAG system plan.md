The system will be run on a cloud GPU with 32–48GB of memory.

Marking the minimal corpus choice as confirmed in the TODOs. I'll update the list now.

Completed: _Confirm allowed retrieval corpora (StatPearls, PubMed, repo corpora)_ (4/23)

Progress: minimal corpus confirmed (StatPearls + MedQA references + Wikipedia). Next I'll produce the full, detailed implementation guide (commands, file layout, hyperparameters, eval scripts, stats).

**Overview**

- **Goal**: Build a RAG pipeline that improves accuracy on MedQA (multiple-choice) vs Qwen3-8B baseline by ≥1%.
- **Assumptions**: Qwen3-8B API for inference; cloud GPU(s) 32–48GB for embedding/reranker; corpora: StatPearls (required — USMLE-aligned clinical summaries), MedQA reference text, Wikipedia medical pages, targeted PubMed abstracts; reranker-only fine-tuning allowed (you may use a pre-finetuned reranker).

**Project Layout**

- **Repo structure** (create under `rag-medqa/`):

```
rag-medqa/
├── configs/
│   ├── default.yaml              # default hyperparams
│   └── experiment_grid.yaml      # sweep grid definition
├── data/
│   ├── raw/                      # downloads (MedQA, StatPearls, wiki)
│   ├── processed/                # cleaned chunks + metadata
│   └── index/                    # FAISS index, BM25 cache, embedding vectors
├── medqa_rag/                    # importable Python package
│   ├── __init__.py
│   ├── config.py                 # dataclass-based config (all hyperparams)
│   ├── data/                     # corpus acquisition + preprocessing (Steps 1–3)
│   │   ├── __init__.py
│   │   ├── download.py           # corpus downloads (StatPearls, Wikipedia, PubMed)
│   │   ├── medqa_loader.py       # MedQA dataset loading + split management
│   │   ├── preprocess.py         # cleaning + chunking
│   │   └── coverage_gate.py      # evidence coverage Go/No-Go check (Step 2b)
│   ├── retrieval/                # embedding, indexing, retrieval (Steps 4–7)
│   │   ├── __init__.py
│   │   ├── base.py               # ABC: BaseRetriever interface
│   │   ├── embedder.py           # compute + cache embeddings
│   │   ├── faiss_index.py        # FAISS build/search (implements BaseRetriever)
│   │   ├── bm25.py               # BM25 sparse retrieval (implements BaseRetriever)
│   │   ├── hybrid.py             # score fusion + dedup (composes retrievers)
│   │   └── query.py              # query formulation strategies
│   ├── rerank/                   # reranking (Step 8)
│   │   ├── __init__.py
│   │   ├── base.py               # ABC: BaseReranker interface
│   │   └── cross_encoder.py      # cross-encoder scoring (supports multiple models)
│   ├── inference/                # LLM prompting (Step 9)
│   │   ├── __init__.py
│   │   ├── llm_client.py         # Qwen3-8B API wrapper
│   │   ├── prompts.py            # prompt templates (direct + CoT)
│   │   └── parser.py             # answer extraction from LLM output
│   ├── evaluation/               # eval + stats (Steps 10–15)
│   │   ├── __init__.py
│   │   ├── harness.py            # run baseline / RAG evaluation loops
│   │   ├── stats.py              # McNemar, bootstrap CI
│   │   └── error_analysis.py     # RAG-help / RAG-hurt breakdown
│   └── experiment.py             # grid sweep orchestration + logging
├── scripts/                      # thin CLI entry points (call into medqa_rag)
│   ├── run_download.py
│   ├── run_preprocess.py
│   ├── run_embed.py
│   ├── run_baseline.py
│   ├── run_rag.py
│   └── run_sweep.py
├── tests/                        # unit + integration tests
│   ├── conftest.py               # shared fixtures (sample questions, mock API)
│   ├── test_preprocess.py
│   ├── test_retrieval.py
│   ├── test_rerank.py
│   ├── test_prompts.py
│   └── test_parser.py
├── results/                      # output logs (gitignored)
├── requirements.txt
├── pyproject.toml                # package metadata + CLI entry points
└── README.md
```

- **Design principles**:
  - `medqa_rag/` is an installable package (`pip install -e .`), not loose scripts. Use `from medqa_rag.retrieval import hybrid`.
  - Subpackages map to pipeline stages: `data/` → Steps 1–3, `retrieval/` → Steps 4–7, `rerank/` → Step 8, `inference/` → Step 9, `evaluation/` → Steps 10–15.
  - Abstract base classes (`base.py`) in `retrieval/` and `rerank/` define swappable interfaces — new retriever or reranker implementations only need to subclass `BaseRetriever` / `BaseReranker`.
  - `config.py` centralizes all hyperparameters as typed dataclasses — no magic numbers scattered across modules.
  - `scripts/` are thin CLI wrappers that parse args and call into the package. They should contain minimal logic.

**Environment**

- **Python**: 3.10+ recommended.
- **Example `requirements.txt`**:
  - sentence-transformers (includes CrossEncoder)
  - transformers
  - faiss-gpu (or faiss-cpu)
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - rank_bm25
  - requests
  - wikipedia-api
  - scispacy
  - https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz # biomedical NER
  - statsmodels
  - tqdm
  - pyyaml
  - pytest
- **`pyproject.toml`** (package metadata):

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "medqa-rag"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []  # list in requirements.txt for pinned versions

[project.scripts]
medqa-download   = "scripts.run_download:main"
medqa-preprocess = "scripts.run_preprocess:main"
medqa-embed      = "scripts.run_embed:main"
medqa-baseline   = "scripts.run_baseline:main"
medqa-rag        = "scripts.run_rag:main"
medqa-sweep      = "scripts.run_sweep:main"
```

- **Install**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # install medqa_rag as editable package
```

- **Run tests**:

```bash
pytest tests/ -v
```

**Configuration System (`medqa_rag/config.py`)**

All hyperparameters are centralized in typed dataclasses. Modules import the config they need — no magic numbers in function signatures.

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml

class QueryFormulation(Enum):
    QUESTION_ONLY = "question_only"
    QUESTION_PLUS_OPTIONS = "question_plus_options"
    ENTITY_QUERY = "entity_query"

class PromptMode(Enum):
    DIRECT = "direct"
    COT = "cot"

class ScoreNormalization(Enum):
    ZSCORE = "z-score"
    MINMAX = "min-max"

@dataclass
class ChunkConfig:
    max_tokens: int = 240
    overlap: int = 50

@dataclass
class RetrievalConfig:
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    dense_k: int = 100
    bm25_k: int = 100
    fusion_alpha: float = 0.5
    score_normalization: ScoreNormalization = ScoreNormalization.ZSCORE
    query_formulation: QueryFormulation = QueryFormulation.QUESTION_ONLY

@dataclass
class RerankConfig:
    model: str = "ncbi/MedCPT-Cross-Encoder"
    enabled: bool = True
    batch_size: int = 16

@dataclass
class InferenceConfig:
    prompt_mode: PromptMode = PromptMode.DIRECT
    top_k_passages: int = 3
    passage_max_tokens: int = 250
    temperature: float = 0.0
    max_tokens: int = 8  # 512 for CoT

@dataclass
class PipelineConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))  # nested parsing omitted for brevity
```

Usage in any module: `from medqa_rag.config import PipelineConfig, RetrievalConfig`.

**Abstract Base Classes**

Swappable retriever and reranker interfaces live in `base.py` within their subpackages. New implementations only need to subclass and implement the abstract methods.

```python
# medqa_rag/retrieval/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    source: str

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return top-k results for the given query."""
        ...

# medqa_rag/rerank/base.py
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Re-score and re-order candidates, return top-k."""
        ...
```

Concrete implementations:

- `faiss_index.py` subclasses `BaseRetriever` — dense retrieval via FAISS.
- `bm25.py` subclasses `BaseRetriever` — sparse retrieval via BM25.
- `hybrid.py` composes multiple `BaseRetriever` instances with score fusion.
- `cross_encoder.py` subclasses `BaseReranker` — cross-encoder scoring.

This makes the experiment grid (Step 12) trivial to implement: instantiate different retriever/reranker combinations by name from config.

**Step 1 — Prepare MedQA**

- **Download**: clone `https://github.com/8023looker/Med-RR` and run their download script per README; save dataset to `data/raw/medqa/`.
- **Verify format**: ensure MedQA examples have fields `id`, `question`, `options` (A-D), and `answer` (single letter).
- **Create splits**: use provided splits or create `dev`/`test`. Keep `dev` for tuning and `test` only for final evaluation.

**Step 2 — Build minimal corpus**

- **MedQA reference text**: extract any reference/explanation texts bundled in MedQA and save to `data/raw/references/`.
- **StatPearls (required)**:
  - StatPearls is a USMLE-oriented clinical reference and is the single most important corpus for MedQA. Its content directly mirrors the clinical reasoning style tested by USMLE questions.
  - Download the StatPearls open-access bulk dataset from NCBI Bookshelf (`https://ftp.ncbi.nlm.nih.gov/pub/litarch/`) or use the StatPearls NBK XML dump. Save to `data/raw/statpearls/`.
  - If bulk access is truly unavailable, treat this as a **blocking risk** and compensate by substantially increasing PubMed abstract coverage (see below).
- **Wikipedia medical pages**:
  - Approach: for each MedQA question, extract **medical entities** (not just nouns) using a biomedical NER model (e.g., `scispacy` with `en_core_sci_md` or `en_ner_bc5cdr_md`) or at minimum multi-word noun-phrase chunking. Query `wikipedia-api` for top N pages (N=5) per entity. Save the page title + text.
  - Example snippet:

```python
# medqa_rag/data/download.py
import spacy, wikipediaapi
nlp = spacy.load("en_ner_bc5cdr_md")  # biomedical NER
wiki = wikipediaapi.Wikipedia('en')

def extract_medical_entities(question_text: str) -> list[str]:
    doc = nlp(question_text)
    return list(set(ent.text for ent in doc.ents))

def fetch_wiki_pages(entities: list[str], top_n: int = 5) -> dict[str, str]:
    pages = {}
    for entity in entities:
        page = wiki.page(entity)
        if page.exists():
            pages[entity] = page.text
    return pages
```

- **Targeted PubMed abstracts (recommended)**:
  - Use NCBI Entrez E-utilities to fetch top-100 abstracts per MedQA topic (safe & free).
  - Respect rate limits (max 3 req/s without API key, 10 req/s with); store as JSONL.
  - PubMed abstracts are especially valuable for questions involving recent treatment guidelines or drug interactions.

**Step 2b — Evidence Coverage Gate (Go/No-Go Checkpoint)**

Before proceeding to Step 3, validate that the corpus can support the downstream goal:

- Sample 200 questions from the dev set.
- For each, manually (or with LLM-assisted labeling) identify ≥1 gold-supporting passage in the corpus.
- Compute **Evidence Coverage@corpus** = fraction of questions with at least one relevant passage found.
- **Go threshold**: Coverage ≥ 60%. If below, expand the corpus (add StatPearls, increase PubMed depth) before continuing.
- This prevents investing in pipeline engineering on a corpus that structurally cannot support ≥1% improvement.

**Step 3 — Clean and chunk**

- **Chunk strategy**:
  - Tokenizer: use the embedding model's tokenizer (match the selected model from Step 4).
  - Target chunk length: 180–256 tokens; overlap: 40–60 tokens.
  - Preserve provenance: store `doc_id`, `source`, `title`, `chunk_id`, `char_start`, `char_end`.
- **Chunking code (concept)**:

```python
# medqa_rag/data/preprocess.py
from transformers import AutoTokenizer
from medqa_rag.config import ChunkConfig

def make_chunker(cfg: ChunkConfig, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def chunk_text(text: str) -> list[str]:
        ids = tokenizer.encode(text)
        chunks = []
        step = cfg.max_tokens - cfg.overlap
        for i in range(0, max(1, len(ids)), step):
            chunk_tokens = ids[i:i + cfg.max_tokens]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        return chunks
    return chunk_text
```

- **Output**: write `data/processed/chunks.jsonl` entries: `{id, chunk_text, doc_id, source, chunk_id}`.

**Step 4 — Embeddings**

- **Model candidates** (priority order — biomedical first):
  1. `pritamdeka/S-PubMedBert-MS-MARCO` (768-d) — PubMedBERT fine-tuned on MS-MARCO for retrieval; best domain/retrieval balance (recommended start)
  2. `FremyCompany/BioLORD-2023` (768-d) — biomedical ontology-grounded embeddings
  3. `multi-qa-mpnet-base-dot-v1` (768-d) — strong general-domain fallback
  4. `all-MiniLM-L6-v2` (384-d) — fast baseline for sanity checks only
- **Rationale**: General-domain models (MiniLM, mpnet) suffer from medical vocabulary mismatch (e.g., "MI" → Michigan vs myocardial infarction; "discharge" → release vs clinical discharge). Biomedical embeddings typically yield 10–20% higher Recall@100 on BioASQ/PubMedQA benchmarks.
- **Embedding code**:

```python
# medqa_rag/retrieval/embedder.py
from sentence_transformers import SentenceTransformer
from medqa_rag.config import RetrievalConfig

class Embedder:
    def __init__(self, cfg: RetrievalConfig):
        self.model = SentenceTransformer(cfg.embedding_model, device='cuda')

    def encode(self, texts: list[str], batch_size: int = 64):
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=True, convert_to_numpy=True)
```

- **Storage**: save emb vectors as `data/index/chunks.npy` and metadata `data/index/chunks_meta.jsonl`.

**Step 5 — Vector store (FAISS PoC)**

- **FAISS options**:
  - Small corpus (<200k vectors): `IndexHNSWFlat` or `IndexFlatIP` with normalized vectors.
  - Medium corpus: `IndexIVFFlat` + `IndexPQ` or `IndexIVFPQ` for compressed storage.
- **Build index (HNSW example)**:

```python
# medqa_rag/retrieval/faiss_index.py
import faiss, numpy as np
from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult

class FaissRetriever(BaseRetriever):
    def __init__(self, index_path: str, meta_path: str, embedder):
        self.index = faiss.read_index(index_path)
        self.meta = ...  # load from meta_path
        self.embedder = embedder

    @classmethod
    def build(cls, vecs_path: str, out_path: str):
        vecs = np.load(vecs_path).astype('float32')
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.add(vecs)
        faiss.write_index(index, out_path)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        q = self.embedder.encode([query])
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k=top_k)
        return [RetrievalResult(chunk_id=str(i), score=float(d),
                text=self.meta[i]['text'], source=self.meta[i]['source'])
                for d, i in zip(D[0], I[0])]
```

**Step 6 — Sparse retrieval (BM25)**

- **BM25 PoC**: use `rank_bm25` over the same `chunks.jsonl`.
- **Combine**: compute BM25 top-K and dense top-K, then union (or score-fuse).
- **Example**:

```python
# medqa_rag/retrieval/bm25.py
from rank_bm25 import BM25Okapi
from medqa_rag.retrieval.base import BaseRetriever, RetrievalResult

class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c['chunk_text'].split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        scores = self.bm25.get_scores(query.split())
        top_idx = scores.argsort()[-top_k:][::-1]
        return [RetrievalResult(chunk_id=str(i), score=float(scores[i]),
                text=self.chunks[i]['chunk_text'], source=self.chunks[i]['source'])
                for i in top_idx]
```

**Step 7 — Hybrid retriever**

- **Algorithm**:
  - Dense top-Nd (e.g., 100) from FAISS
  - BM25 top-Ns (e.g., 100)
  - Union → deduplicate → pass union to reranker
- **Query formulation**: the retrieval query should be constructed carefully, not just the raw question text. Experiment with:
  - `question_only`: raw question text
  - `question_plus_options`: concatenate question + all options (provides more semantic signal, but adds noise)
  - `entity_query`: extract biomedical entities (from Step 2 NER) and concatenate as query
  - Include query formulation strategy in the experimental grid (Step 12).
- **Scoring fusion**:
  - Normalize dense and BM25 scores then weighted sum; tune weight α ∈ {0.3, 0.5, 0.7}.
  - **Normalization method**: test both z-score and min-max normalization. Z-score is more robust to outliers; min-max preserves score distribution shape. Include normalization method as a grid variable.

**Step 8 — Reranker**

- **Use pre-finetuned reranker** (biomedical preferred):
  1. `ncbi/MedCPT-Cross-Encoder` — trained on PubMed query-article pairs; best domain fit (recommended)
  2. `cross-encoder/ms-marco-MiniLM-L-12-v2` — general-domain fallback, stronger than L-6 variant
  3. `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast general-domain baseline
  - **Rationale**: MS-MARCO rerankers are trained on short web queries, which differ substantially from USMLE-style clinical vignettes. A biomedical cross-encoder better calibrates relevance for medical evidence passages.
  - Validate each reranker on a small dev set (50–100 questions) before full runs.
- **Rerank code**:

```python
# medqa_rag/rerank/cross_encoder.py
from sentence_transformers import CrossEncoder
from medqa_rag.rerank.base import BaseReranker
from medqa_rag.retrieval.base import RetrievalResult
from medqa_rag.config import RerankConfig

class CrossEncoderReranker(BaseReranker):
    def __init__(self, cfg: RerankConfig):
        self.model = CrossEncoder(cfg.model, device='cuda')
        self.batch_size = cfg.batch_size

    def rerank(self, query: str, candidates: list[RetrievalResult],
               top_k: int) -> list[RetrievalResult]:
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]
```

- **Optimization**: batch scoring on GPU, use fp16; convert model to ONNX for faster inference if needed.

**Step 9 — Prompting Qwen3-8B (multiple-choice)**

- **Prompt design rules**:
  - Provide only the top-K evidence passages (K=3–5) after reranking.
  - **Truncate each passage to ≤ 250 tokens** to control total prompt length (include `passage_max_tokens` as a configurable parameter).
  - Include provenance labels `[1]`, `[2]` with source id and short citation.
  - Use temperature = 0.0 to reduce randomness.
  - **Two prompt modes** (include both in experimental grid):
    - **Direct mode**: ask for single-letter output only; set `max_tokens=8`.
    - **CoT (Chain-of-Thought) mode**: ask the model to reason step-by-step before answering; set `max_tokens=512`. Parse the final letter from response.
  - **Rationale for CoT**: For 8B-class models on USMLE-style clinical reasoning, CoT prompting typically yields 3–8% accuracy gains by allowing the model to decompose multi-step diagnostic logic. This is a high-leverage variable that should not be omitted.

- **Example prompt template (Direct mode)**:

```
You are a medical assistant. Use the evidence below to choose the single best option. Output ONLY the single letter (A/B/C/D).

Question: {question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}

Evidence:
[1] {passage1} (Source: {source1})
[2] {passage2} (Source: {source2})
[3] {passage3} (Source: {source3})

Answer (letter only):
```

- **Example prompt template (CoT mode)**:

```
You are a medical assistant. Use the evidence below to answer the question. Think step by step: first identify the key clinical findings, then reason through each option using the evidence, and finally choose the single best answer.

Question: {question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}

Evidence:
[1] {passage1} (Source: {source1})
[2] {passage2} (Source: {source2})
[3] {passage3} (Source: {source3})

Step-by-step reasoning:
```

- **Parsing** (`medqa_rag/inference/parser.py`): For Direct mode, parse the first non-empty letter [A-D]. For CoT mode, extract the final letter from the response using regex `r'[Aa]nswer.*?([A-D])'` or take the last occurrence of [A-D].
- **API call** (`medqa_rag/inference/llm_client.py`): set `temperature=0.0`, adjust `max_tokens` per prompt mode (read from `InferenceConfig`).
- **Prompt templates** live in `medqa_rag/inference/prompts.py` — separated from the API client so templates can be tested and iterated without touching network code.

**Step 10 — Baseline evaluation**

- **Baseline harness**:
  - For each MedQA question, send prompt containing only question + options (no evidence).
  - Record `baseline_pred`, `logprob` (if API provides), and response text.
  - Save logs to `results/baseline_dev.jsonl`.
- **Accuracy**: compute simple match fraction.

**Step 11 — RAG evaluation**

- **Pipeline**:
  - For each question:
    - normalize question → retrieve hybrid top candidates → rerank → select top-K passages.
    - build prompt with passages → call Qwen → parse letter → record `rag_pred`.
  - Save `results/rag_dev.jsonl` with retrieved passage ids & scores.
- **Run on dev set** first for hyperparameter tuning; final run on test set only once.

**Step 12 — Experimental sweep (suggested grid)**

- **Grid**:
  - `embedding`: S-PubMedBert-MS-MARCO (768), BioLORD-2023 (768), mpnet (768)
  - `dense_k`: {50, 100}
  - `bm25_k`: {0, 50, 100}
  - `fusion_alpha`: {0.3, 0.5, 0.7}
  - `score_normalization`: {z-score, min-max}
  - `query_formulation`: {question_only, question_plus_options, entity_query}
  - `reranker`: {none, MedCPT-Cross-Encoder, ms-marco-MiniLM-L-12-v2}
  - `top_K_passages` for prompt: {3, 5}
  - `passage_max_tokens`: {200, 300}
  - `prompt_mode`: {direct, cot}
  - `temperature`: {0.0}
- **Priority** (sweep in this order to manage compute budget):
  1. **prompt_mode** {direct, cot} × **reranker** {none, MedCPT} — highest expected impact; run first on 200-question dev subset.
  2. **embedding** × **query_formulation** — determines retrieval quality ceiling.
  3. **top_K_passages** × **passage_max_tokens** — controls evidence density vs noise.
  4. **fusion_alpha** × **bm25_k** — fine-tuning the hybrid mix.
- **Compute note**: Full grid has many combinations. Use the priority order above with early stopping: if a variable shows <0.3% effect on the dev subset, fix it at the best value and move on.

**Step 13 — Logging and reproducibility**

- **Log format**: JSONL rows with `id, question, options, answer, baseline_pred, rag_pred, retrieved_ids, reranker_scores, prompt`.
- **Repro**: set `PYTHONHASHSEED`, seed any sampling, freeze package versions (`pip freeze > requirements.lock`), provide Dockerfile.

**Step 14 — Statistical testing**

- **Paired significance** (preferred for paired classifiers):
  - Use McNemar's test on binary correctness pairs (baseline correct vs RAG correct).
  - Example (Python):

```python
# medqa_rag/evaluation/stats.py
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar(n00: int, n01: int, n10: int, n11: int) -> float:
    """n00=both wrong, n01=baseline wrong+RAG correct,
       n10=baseline correct+RAG wrong, n11=both correct."""
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue
```

- **Interpretation**: p < 0.05 → significant difference.
- **A priori power analysis**:
  - For McNemar's test detecting a 1% absolute accuracy difference (~13 questions out of 1273), statistical power depends on the number of discordant pairs (where baseline and RAG disagree).
  - At typical baseline ~55% accuracy, expect ~35–40% discordant pairs → ~450–500 discordant pairs.
  - A 1% absolute difference implies the discordant split shifts by ~13 questions → effect size is small.
  - **Expected power at n=1273: ~0.35–0.50** — marginal. Mitigation strategies:
    - Use **one-sided** McNemar test (H₁: RAG > baseline), which increases power by ~10–15%.
    - Supplement with bootstrap CI (below) which can detect directional improvement even when McNemar is underpowered.
    - If possible, evaluate on the full MedQA test set (not a subset) to maximize sample size.
  - If the achieved improvement is >2%, power becomes adequate (~0.75+). The power concern is specific to the ≥1% boundary.
- **Bootstrap CI for accuracy difference**:

```python
# medqa_rag/evaluation/stats.py (continued)
import numpy as np

def bootstrap_diff(baseline_correct: np.ndarray, rag_correct: np.ndarray,
                   n: int = 10000) -> tuple[float, float]:
    """Return 95% CI for (rag_accuracy - baseline_accuracy)."""
    n_examples = len(baseline_correct)
    diffs = []
    for _ in range(n):
        idx = np.random.randint(0, n_examples, n_examples)
        diffs.append(rag_correct[idx].mean() - baseline_correct[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)
```

- **Decision rule**: If RAG accuracy - baseline accuracy ≥ 1% and p-value < 0.05 (or bootstrap lower CI > 0%), accept improvement.

**Step 15 — Error analysis**

- **Collect**:
  - Cases where RAG improved but baseline failed.
  - Cases where RAG failed but baseline correct.
  - For each, inspect retrieved passages and reranker scores.
- **Metrics**:
  - Hallucination cases (asserted facts not in retrieved evidence).
  - Evidence coverage: fraction of questions whose gold-supporting passage appears in top-N retrieval.
  - **RAG-hurt analysis**: specifically examine cases where baseline was correct but RAG was wrong. Common causes:
    - Retrieved passages contain plausible but incorrect information → misleads the model.
    - Too many passages dilute the signal → model ignores correct evidence.
    - CoT reasoning goes astray due to noisy evidence.
  - Use RAG-hurt rate to set an upper bound on acceptable noise: if RAG-hurt > RAG-help, the configuration is net negative.

**Step 16 — Performance & cost estimates**

- **Embeddings**: ~2–20 minutes for 100k chunks on single 32GB GPU (depends on batch size).
- **Reranker**: GPU inference; scoring 100 candidates per query on reranker ~0.2–1s/query depending on batch and model.
- **API cost**: depends on Qwen pricing — minimize tokens in prompts and retrieved content.
- **Disk**:
  - 1M vectors @768-d float32 ≈ 3.0 GB; quantization (PQ/SQ) reduces to ~0.5–1.0 GB.

**Step 17 — Practical tips**

- **Start small**: run full pipeline on `dev` subset (~500 Qs) to iterate fast.
- **Limit passage length** in prompt—truncate to 250–300 tokens per passage.
- **Cache** retrieval results during grid search to avoid repeated indexing/api costs.
- **Use deterministic prompt parsing**: require "Answer: A" or only the letter.

**Step 18 — Final acceptance**

- **Stop** when:
  - Dev-tuned RAG yields ≥1% absolute accuracy improvement on dev.
  - On test set, improvement ≥1% and McNemar p<0.05 (or bootstrap CI excludes 0).
- **Deliverables**:
  - `results/` JSONL logs for baseline and best RAG run.
  - `configs/best.yaml` with hyperparams.
  - `reports/final_report.md` summarizing metrics, stats, and error analysis.

**Appendix A — Example commands**

- Clone Med-RR and download:

```bash
git clone https://github.com/8023looker/Med-RR
cd Med-RR
# follow their README to download MedQA into data/raw/medqa
```

- Install package and run pipeline (using CLI entry points from pyproject.toml):

```bash
pip install -e .
medqa-download --corpus statpearls wiki pubmed --out data/raw/
medqa-preprocess --input data/raw/ --out data/processed/chunks.jsonl --config configs/default.yaml
medqa-embed --chunks data/processed/chunks.jsonl --config configs/default.yaml --out data/index/
```

- Or run via scripts directly:

```bash
python scripts/run_preprocess.py --input data/raw/ --out data/processed/chunks.jsonl
python scripts/run_embed.py --chunks data/processed/chunks.jsonl --out data/index/chunks.npy
```

- Run baseline eval:

```bash
medqa-baseline --medqa data/raw/medqa/dev.jsonl --out results/baseline_dev.jsonl
```

- Run RAG eval:

```bash
medqa-rag --config configs/default.yaml --medqa data/raw/medqa/dev.jsonl --out results/rag_dev.jsonl
```

- Run experiment sweep:

```bash
medqa-sweep --grid configs/experiment_grid.yaml --medqa data/raw/medqa/dev.jsonl --out results/sweep/
```

- Compute stats (from Python):

```bash
python -c "from medqa_rag.evaluation.stats import run_mcnemar; ..."
# or use the harness:
python scripts/run_baseline.py --mode stats --baseline results/baseline_dev.jsonl --rag results/best_rag_dev.jsonl
```

**Appendix B — Testing strategy**

- **Unit tests** (`tests/`): test individual components in isolation.
  - `test_preprocess.py` — verify chunking produces correct token counts, overlap, and provenance metadata.
  - `test_retrieval.py` — verify `BM25Retriever` and `FaissRetriever` return correct `RetrievalResult` format; test `HybridRetriever` score fusion with mock retrievers.
  - `test_rerank.py` — verify `CrossEncoderReranker` re-orders candidates correctly (mock the model).
  - `test_prompts.py` — verify prompt templates render correctly for both direct and CoT modes.
  - `test_parser.py` — verify answer extraction handles edge cases (multi-letter output, CoT reasoning, empty responses).
- **Integration tests**: run a mini pipeline (5 questions, small corpus) end-to-end to verify the full retrieval → rerank → inference → evaluation chain.
- **Fixtures** (`conftest.py`): provide sample MedQA questions, mock LLM responses, small chunk corpora.
- **Run**: `pytest tests/ -v` (add `-x` for fail-fast during development).
