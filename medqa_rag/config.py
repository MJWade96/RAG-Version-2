from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin
import types
import yaml


class QueryFormulation(str, Enum):
    QUESTION_ONLY = "question_only"
    QUESTION_PLUS_OPTIONS = "question_plus_options"
    ENTITY_QUERY = "entity_query"


class PromptMode(str, Enum):
    DIRECT = "direct"
    COT = "cot"


class ScoreNormalization(str, Enum):
    ZSCORE = "z-score"
    MINMAX = "min-max"


@dataclass(slots=True)
class ChunkConfig:
    max_tokens: int = 240
    overlap: int = 50
    min_chunk_chars: int = 50


@dataclass(slots=True)
class RetrievalConfig:
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_device: str = "cpu"
    dense_k: int = 100
    bm25_k: int = 100
    fusion_alpha: float = 0.5
    score_normalization: ScoreNormalization = ScoreNormalization.ZSCORE
    query_formulation: QueryFormulation = QueryFormulation.QUESTION_ONLY


@dataclass(slots=True)
class RerankConfig:
    model: str = "ncbi/MedCPT-Cross-Encoder"
    enabled: bool = True
    batch_size: int = 16
    top_k: int = 20


@dataclass(slots=True)
class InferenceConfig:
    prompt_mode: PromptMode = PromptMode.DIRECT
    top_k_passages: int = 3
    passage_max_tokens: int = 250
    temperature: float = 0.0
    enable_thinking: bool = True
    timeout: int = 60
    max_workers: int = 1
    rate_limit: float = 10.0  # requests per second


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    subset_size: int | None = None


@dataclass(slots=True)
class PipelineConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return _build_dataclass(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["results_dir"] = str(self.results_dir)
        payload["retrieval"]["score_normalization"] = self.retrieval.score_normalization.value
        payload["retrieval"]["query_formulation"] = self.retrieval.query_formulation.value
        payload["inference"]["prompt_mode"] = self.inference.prompt_mode.value
        payload["inference"]["timeout"] = self.inference.timeout
        payload["inference"]["max_workers"] = self.inference.max_workers
        payload["inference"]["rate_limit"] = self.inference.rate_limit
        return payload


T = TypeVar("T")


def _build_dataclass(cls: type[T], data: dict[str, Any] | None) -> T:
    payload = data or {}
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name in payload:
            raw_value = payload[field_info.name]
        elif field_info.default is not MISSING:
            raw_value = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            raw_value = field_info.default_factory()
        else:  # pragma: no cover - not used in current config dataclasses
            raise ValueError(f"Missing required config field: {field_info.name}")
        kwargs[field_info.name] = _convert_value(field_info.type, raw_value)
    return cls(**kwargs)


def _convert_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        return _convert_value(non_none[0], value) if non_none else value

    if origin is list:
        subtype = args[0] if args else Any
        return [_convert_value(subtype, item) for item in value]

    if origin is tuple:
        subtype = args[0] if args else Any
        return tuple(_convert_value(subtype, item) for item in value)

    if origin is dict:
        key_type, val_type = args if len(args) == 2 else (Any, Any)
        return {
            _convert_value(key_type, key): _convert_value(val_type, item)
            for key, item in value.items()
        }

    if isinstance(field_type, type):
        if issubclass(field_type, Enum):
            return field_type(value)
        if field_type is Path:
            return Path(value)
        if is_dataclass(field_type):
            return _build_dataclass(field_type, value)
    return value


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    return PipelineConfig.from_yaml(path)
