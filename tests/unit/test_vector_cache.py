import hashlib
import math

from specinfer.cache.vector_cache import SemanticCache


class HashEmbedder:
    """Deterministic bag-of-words embedder: similar texts -> close vectors."""

    dim = 64

    def embed(self, text):
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def test_hit_and_miss():
    c = SemanticCache(HashEmbedder(), threshold=0.05)
    assert c.lookup("anything") is None
    c.store("what is the capital of france", "Paris")
    hit = c.lookup("what is the capital of france")
    assert hit and hit.completion == "Paris" and hit.distance < 1e-6
    assert c.lookup("explain quicksort with complexity analysis") is None
    assert c.count() == 1


def test_persistent_path(tmp_path):
    p = str(tmp_path / "q")
    c = SemanticCache(HashEmbedder(), path=p)
    c.store("hello world", "hi")
    c.client.close()
    c2 = SemanticCache(HashEmbedder(), path=p)
    assert c2.lookup("hello world").completion == "hi"
