from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be configured.")
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


def save_embeddings(vectors: np.ndarray, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, vectors)
    return target


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(Path(path))
