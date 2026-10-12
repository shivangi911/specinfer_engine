"""Local prompt embedding pipeline (nomic-embed-text by default, 768-d)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class NomicEmbedder:
    def __init__(self, name: str = "nomic-ai/nomic-embed-text-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, trust_remote_code=True)
        self.dim = int(self.model.get_sentence_embedding_dimension() or 768)

    def embed(self, text: str) -> list[float]:
        v = self.model.encode(("search_query: " + text) if "nomic" in self.name else text, normalize_embeddings=True)
        return np.asarray(v, dtype=float).tolist()
