"""KV synchronizer + speculative loop against a deterministic fake backend (no downloads)."""
import torch

from specinfer.engine.speculative import baseline_generate, speculative_generate

V = 11


class FakeLM:
    """Next-token logits depend on the previous token and a salt; tracks cache length."""

    def __init__(self, salt: int):
        self.salt, self._toks = salt, []

    @property
    def length(self):
        return len(self._toks)

    def forward(self, tokens):
        rows = []
        for t in tokens:
            self._toks.append(t)
            g = torch.Generator().manual_seed(t * 31 + self.salt * (t % 2))
            rows.append(torch.randn(V, generator=g) * 3)
        return torch.stack(rows)

    def truncate(self, n):
        self._toks = self._toks[:n]

    def reset(self):
        self._toks = []


def test_greedy_speculative_equals_baseline():
    prompt = [1, 2, 3]
    base = [t for c in baseline_generate(FakeLM(1), prompt, set(), 40) for t in c]
    spec = [
        t for c in speculative_generate(FakeLM(2), FakeLM(1), prompt, set(), 40, k=4) for t in c
    ]
    assert spec == base


def test_caches_stay_synced_with_sampling():
    d, t = FakeLM(2), FakeLM(1)
    out = [x for c in speculative_generate(d, t, [5], set(), 30, 3, 1.0, seed=0) for x in c]
    assert len(out) == 30
    assert d.length == t.length
