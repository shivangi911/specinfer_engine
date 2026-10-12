"""FastAPI proxy gateway. Run: python -m specinfer.server.app"""
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI

from specinfer.config import settings
from specinfer.engine.service import Metrics, TextGenerator
from specinfer.server.routes import router


def create_app(generator: TextGenerator, cache=None, generator_metrics: Metrics | None = None
               ) -> FastAPI:
    app = FastAPI(title="SpecInfer-Engine")
    app.state.generator = generator
    app.state.cache = cache
    app.state.generator_metrics = generator_metrics or getattr(generator, "metrics", Metrics())
    app.state.metrics = SimpleNamespace(requests=0, cache_hits=0)
    app.include_router(router)
    return app


def build_default_app() -> FastAPI:
    from specinfer.engine.model_loader import load_backend
    from specinfer.engine.service import SpecService

    draft = load_backend(settings.backend, settings.draft_model)
    target = load_backend(settings.backend, settings.target_model)
    gen = SpecService(draft, target, target.tokenizer, target.eos_ids,  # type: ignore[attr-defined]
                      k=settings.lookahead_k, model_name=settings.target_model)
    cache = None
    if settings.cache_enabled:
        from specinfer.cache.embeddings import NomicEmbedder
        from specinfer.cache.vector_cache import SemanticCache

        try:
            embedder = NomicEmbedder(settings.embed_model)
        except Exception as e:  # noqa: BLE001  # nomic needs remote code; fall back to a small local model
            print(f"[specinfer] {settings.embed_model} unavailable ({e!r}); using MiniLM")
            embedder = NomicEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        cache = SemanticCache(embedder, settings.cache_path, settings.cache_distance_threshold)
    return create_app(gen, cache)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_default_app(), host=settings.host, port=settings.port)
