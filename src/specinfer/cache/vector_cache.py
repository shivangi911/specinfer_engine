"""Semantic completion cache on embedded Qdrant (cosine distance)."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from specinfer.cache.embeddings import Embedder

COLLECTION = "completions"


@dataclass
class CacheHit:
    completion: str
    distance: float


class SemanticCache:
    def __init__(self, embedder: Embedder, path: str | None = None, threshold: float = 0.05):
        """path=None uses an in-memory index (tests)."""
        self.embedder = embedder
        self.threshold = threshold
        self._lock = threading.Lock()
        self.client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE)
            )

    def lookup(self, prompt: str) -> CacheHit | None:
        vec = self.embedder.embed(prompt)
        with self._lock:
            res = self.client.query_points(COLLECTION, query=vec, limit=1, with_payload=True)
        if not res.points:
            return None
        top = res.points[0]
        dist = 1.0 - float(top.score)  # qdrant cosine score is similarity
        if dist < self.threshold and top.payload:
            return CacheHit(str(top.payload["completion"]), dist)
        return None

    def store(self, prompt: str, completion: str) -> None:
        vec = self.embedder.embed(prompt)
        with self._lock:
            self.client.upsert(
                COLLECTION,
                [PointStruct(id=str(uuid.uuid4()), vector=vec,
                             payload={"prompt": prompt, "completion": completion})],
            )

    def count(self) -> int:
        return self.client.count(COLLECTION).count
