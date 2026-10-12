"""High-level text generation service wrapping the draft/target pair."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from specinfer.engine.model_loader import ModelBackend
from specinfer.engine.speculative import GenStats, baseline_generate, speculative_generate


@dataclass
class Metrics:
    requests: int = 0
    cache_hits: int = 0
    drafted: int = 0
    accepted: int = 0
    tokens: int = 0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "requests": self.requests,
                "cache_hits": self.cache_hits,
                "tokens": self.tokens,
                "acceptance_rate": self.accepted / self.drafted if self.drafted else 0.0,
                "tok_per_s": self.tokens / self.seconds if self.seconds else 0.0,
            }


class TextGenerator(Protocol):
    model_name: str

    def stream(
        self, messages: list[dict], temperature: float, top_p: float, max_tokens: int,
        seed: int | None = None,
    ) -> Iterator[str]: ...


class SpecService:
    """Single-flight (one generation at a time) speculative generator."""

    def __init__(self, draft: ModelBackend, target: ModelBackend, tokenizer, eos_ids: set[int],
                 k: int = 4, speculative: bool = True, metrics: Metrics | None = None,
                 model_name: str = "specinfer"):
        self.draft, self.target, self.tok, self.eos_ids = draft, target, tokenizer, eos_ids
        self.k, self.speculative = k, speculative
        self.metrics = metrics or Metrics()
        self.model_name = model_name
        self._lock = threading.Lock()

    def encode_chat(self, messages: list[dict]) -> list[int]:
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        return list(out["input_ids"])

    def stream(self, messages, temperature=0.0, top_p=1.0, max_tokens=256, seed=None):
        prompt = self.encode_chat(messages)
        st = GenStats()
        with self._lock:
            if self.speculative:
                gen = speculative_generate(self.draft, self.target, prompt, self.eos_ids,
                                           max_tokens, self.k, temperature, top_p, seed, st)
            else:
                gen = baseline_generate(self.target, prompt, self.eos_ids, max_tokens,
                                        temperature, top_p, seed, st)
            ids: list[int] = []
            emitted = ""
            try:
                for chunk in gen:
                    ids += [t for t in chunk if t not in self.eos_ids]
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    if text.endswith("\ufffd"):  # incomplete multi-byte char
                        continue
                    if len(text) > len(emitted):
                        yield text[len(emitted):]
                        emitted = text
                text = self.tok.decode(ids, skip_special_tokens=True)
                if len(text) > len(emitted):
                    yield text[len(emitted):]
            finally:
                m = self.metrics
                with m.lock:
                    m.drafted += st.drafted
                    m.accepted += st.accepted
                    m.tokens += st.tokens
                    m.seconds += st.seconds
