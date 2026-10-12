"""KV-cache synchronization between draft and target models.

Invariant between cycles: with `seq` = prompt + committed tokens, BOTH caches hold
exactly seq[:-1] (the last committed token is the pending input of the next cycle).
"""
from __future__ import annotations

from specinfer.engine.model_loader import ModelBackend


class KVCacheSynchronizer:
    def __init__(self, draft: ModelBackend, target: ModelBackend):
        self.draft = draft
        self.target = target

    def prefill(self, prompt: list[int]) -> None:
        self.draft.reset()
        self.target.reset()
        if len(prompt) > 1:
            self.draft.forward(prompt[:-1])
            self.target.forward(prompt[:-1])
        self.assert_synced()

    def commit(self, base: int, n_accepted: int, k: int, draft_tokens: list[int]) -> None:
        """After verifying k drafts from a cache base length `base` and accepting n.

        Target cache is base+k+1 -> truncate to base+n+1.
        Draft cache is base+k (last draft never fed): truncate on rejection, or feed the
        last draft token when everything was accepted.
        """
        new_len = base + n_accepted + 1
        self.target.truncate(new_len)
        if n_accepted == k and k > 0:
            self.draft.forward([draft_tokens[-1]])
        else:
            self.draft.truncate(new_len)
        self.assert_synced()

    def assert_synced(self) -> None:
        assert self.draft.length == self.target.length, (self.draft.length, self.target.length)
