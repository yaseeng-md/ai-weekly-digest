from __future__ import annotations

from functools import cached_property
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    @cached_property
    def model(self) -> SentenceTransformer:
        # CPU is the default so embedding inference does not compete with the local LLM for limited VRAM.
        return SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        text = text.strip() or "(empty article)"
        vector = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32).tolist()

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            [text.strip() or "(empty article)" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()
