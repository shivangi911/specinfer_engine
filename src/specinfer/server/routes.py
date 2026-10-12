"""OpenAI-compatible /v1/chat/completions with SSE streaming and semantic cache."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "specinfer"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    seed: int | None = None


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> str:
    body = {
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _cache_key(req: ChatRequest) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in req.messages)


def _token_stream(req: ChatRequest, request: Request):
    """Sync generator of text pieces; consults the semantic cache first."""
    state = request.app.state
    state.metrics.requests += 1
    cache = state.cache
    cacheable = cache is not None and req.temperature == 0
    key = _cache_key(req)
    if cacheable:
        hit = cache.lookup(key)
        if hit:
            state.metrics.cache_hits += 1
            text = hit.completion
            for i in range(0, len(text), 16):
                yield text[i:i + 16]
            return
    parts: list[str] = []
    msgs = [m.model_dump() for m in req.messages]
    for piece in state.generator.stream(msgs, req.temperature, req.top_p, req.max_tokens, req.seed):
        parts.append(piece)
        yield piece
    if cacheable and parts:
        cache.store(key, "".join(parts))


@router.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, request: Request):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model = req.model
    if req.stream:
        def sse():
            yield _chunk(cid, model, {"role": "assistant"})
            for piece in _token_stream(req, request):
                yield _chunk(cid, model, {"content": piece})
            yield _chunk(cid, model, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")
    text = "".join(_token_stream(req, request))
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
    })


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics")
def metrics(request: Request):
    s = request.app.state
    return {**s.generator_metrics.snapshot(), "requests": s.metrics.requests,
            "cache_hits": s.metrics.cache_hits}
