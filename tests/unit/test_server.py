import json

from fastapi.testclient import TestClient

from specinfer.cache.vector_cache import SemanticCache
from specinfer.server.app import create_app
from tests.unit.test_vector_cache import HashEmbedder


class FakeGen:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def stream(self, messages, temperature, top_p, max_tokens, seed=None):
        self.calls += 1
        yield from ["Hello", ", ", "world"]


def _client():
    gen = FakeGen()
    return gen, TestClient(create_app(gen, SemanticCache(HashEmbedder())))


def test_non_stream_and_cache_hit():
    gen, c = _client()
    body = {"messages": [{"role": "user", "content": "say hello"}]}
    r1 = c.post("/v1/chat/completions", json=body).json()
    r2 = c.post("/v1/chat/completions", json=body).json()
    assert r1["choices"][0]["message"]["content"] == "Hello, world" == \
        r2["choices"][0]["message"]["content"]
    assert gen.calls == 1  # second request served from cache
    assert c.get("/metrics").json()["cache_hits"] == 1


def test_sse_stream_and_sampling_bypasses_cache():
    gen, c = _client()
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "temperature": 0.8}
    for _ in range(2):
        r = c.post("/v1/chat/completions", json=body)
        lines = [ln for ln in r.text.split("\n\n") if ln]
        assert lines[-1] == "data: [DONE]"
        text = "".join(
            json.loads(ln[6:])["choices"][0]["delta"].get("content", "") for ln in lines[:-1]
        )
        assert text == "Hello, world"
    assert gen.calls == 2
