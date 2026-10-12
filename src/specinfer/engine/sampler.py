"""Tensorized speculative rejection sampling (Leviathan et al. / Chen et al.)."""
from __future__ import annotations

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    """Convert logits [..., V] to the *sampling* distribution.

    The same transform must be applied to draft and target for the output to be
    distributed exactly as the target's processed distribution. temperature<=0 gives
    a one-hot (greedy) distribution.
    """
    logits = logits.float()
    if temperature <= 0:
        return torch.nn.functional.one_hot(
            logits.argmax(-1), logits.shape[-1]
        ).to(logits.dtype)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        csum = sorted_p.cumsum(-1)
        drop = (csum - sorted_p) >= top_p  # keep smallest set whose mass >= top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample_from(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def rejection_sample(
    draft_tokens: torch.Tensor,  # [K] long
    q: torch.Tensor,  # [K, V] draft probs the tokens were drawn from
    p: torch.Tensor,  # [K+1, V] target probs (row K is the bonus position)
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Return (n_accepted, next_token).

    Emitted tokens are draft_tokens[:n_accepted] followed by next_token, which is a
    resample from the residual distribution on rejection or a bonus sample from p[K]
    when all K drafts are accepted.
    """
    k = draft_tokens.shape[0]
    if k == 0:
        return 0, sample_from(p[0], generator)
    idx = torch.arange(k)
    ratio = p[idx, draft_tokens] / q[idx, draft_tokens].clamp_min(1e-20)
    u = torch.rand(k, generator=generator)
    accept = (u <= ratio.clamp(max=1.0)).long()
    n = int(accept.cumprod(0).sum().item())  # length of accepted prefix
    if n == k:
        return n, sample_from(p[k], generator)
    residual = (p[n] - q[n]).clamp_min(0)
    total = residual.sum()
    dist = residual / total if total > 0 else p[n]
    return n, sample_from(dist, generator)


if __name__ == "__main__":  # pragma: no cover
    from specinfer.engine.cli import main

    main()
