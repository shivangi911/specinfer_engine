"""Speculative and baseline decoding loops."""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from specinfer.engine.kv_cache import KVCacheSynchronizer
from specinfer.engine.model_loader import ModelBackend
from specinfer.engine.sampler import logits_to_probs, rejection_sample, sample_from


@dataclass
class GenStats:
    tokens: int = 0
    drafted: int = 0
    accepted: int = 0
    cycles: int = 0
    seconds: float = 0.0
    token_ids: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tok_per_s(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


def _trim_eos(tokens: list[int], eos: set[int]) -> tuple[list[int], bool]:
    for i, t in enumerate(tokens):
        if t in eos:
            return tokens[: i + 1], True
    return tokens, False


def speculative_generate(
    draft: ModelBackend,
    target: ModelBackend,
    prompt: list[int],
    eos_ids: set[int],
    max_new_tokens: int = 256,
    k: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    stats: GenStats | None = None,
) -> Iterator[list[int]]:
    """Yield lists of newly verified tokens, one list per draft/verify cycle."""
    st = stats if stats is not None else GenStats()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    sync = KVCacheSynchronizer(draft, target)
    t0 = time.perf_counter()
    sync.prefill(prompt)
    seq = list(prompt)
    produced = 0
    done = False
    while not done and produced < max_new_tokens:
        base = len(seq) - 1
        kk = min(k, max_new_tokens - produced)
        # 1) draft proposes kk tokens autoregressively
        d_tokens: list[int] = []
        q_rows: list[torch.Tensor] = []
        feed = [seq[-1]]
        for _ in range(kk):
            q = logits_to_probs(draft.forward(feed)[-1], temperature, top_p)
            tok = sample_from(q, gen)
            d_tokens.append(tok)
            q_rows.append(q)
            feed = [tok]
        # 2) target verifies all in one forward pass
        p = logits_to_probs(target.forward([seq[-1], *d_tokens]), temperature, top_p)
        n, nxt = rejection_sample(torch.tensor(d_tokens), torch.stack(q_rows), p, gen)
        # 3) sync caches, emit
        sync.commit(base, n, kk, d_tokens)
        new, done = _trim_eos([*d_tokens[:n], nxt], eos_ids)
        new = new[: max_new_tokens - produced]  # bonus token may exceed the budget
        st.drafted += kk
        st.accepted += n
        st.cycles += 1
        seq += new
        produced += len(new)
        st.tokens += len(new)
        st.token_ids += new
        st.seconds = time.perf_counter() - t0
        yield new
    st.seconds = time.perf_counter() - t0


def baseline_generate(
    target: ModelBackend,
    prompt: list[int],
    eos_ids: set[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    stats: GenStats | None = None,
) -> Iterator[list[int]]:
    st = stats if stats is not None else GenStats()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.perf_counter()
    target.reset()
    if len(prompt) > 1:
        target.forward(prompt[:-1])
    last = prompt[-1]
    for _ in range(max_new_tokens):
        probs = logits_to_probs(target.forward([last])[-1], temperature, top_p)
        last = sample_from(probs, gen)
        st.tokens += 1
        st.token_ids.append(last)
        st.seconds = time.perf_counter() - t0
        yield [last]
        if last in eos_ids:
            break
    st.seconds = time.perf_counter() - t0
