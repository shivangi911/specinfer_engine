"""MLX backend for Apple Silicon (mlx-lm). UNTESTED on this Linux dev box.

Install with `pip install -e .[mlx]` on a Mac and run with SPECINFER_BACKEND=mlx, e.g.
SPECINFER_DRAFT_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
SPECINFER_TARGET_MODEL=mlx-community/Qwen2.5-32B-Instruct-4bit
"""
from __future__ import annotations

import numpy as np
import torch


class MLXBackend:
    def __init__(self, name: str):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self._mx = mx
        self._make_cache = make_prompt_cache
        self.name = name
        self.model, self.tokenizer = load(name)
        self.eos_ids: set[int] = {int(i) for i in self.tokenizer.eos_token_ids}
        self._cache = make_prompt_cache(self.model)
        self._len = 0

    @property
    def length(self) -> int:
        return self._len

    def forward(self, tokens: list[int]) -> torch.Tensor:
        mx = self._mx
        logits = self.model(mx.array([tokens]), cache=self._cache)
        self._len += len(tokens)
        arr = np.array(logits[0].astype(mx.float32))
        return torch.from_numpy(arr)

    def truncate(self, n: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache

        drop = self._len - n
        if drop > 0:
            trim_prompt_cache(self._cache, drop)
            self._len = n

    def reset(self) -> None:
        self._cache = self._make_cache(self.model)
        self._len = 0
