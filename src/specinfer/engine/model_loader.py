"""Model backends. Each backend owns its KV cache behind a tiny interface."""
from __future__ import annotations

from typing import Any, Protocol

import torch


class ModelBackend(Protocol):
    """Stateful causal LM. `forward` appends tokens to the internal KV cache."""

    @property
    def length(self) -> int: ...

    def forward(self, tokens: list[int]) -> torch.Tensor:
        """Feed tokens, return logits [len(tokens), V] (row i predicts token after tokens[i])."""

    def truncate(self, n: int) -> None:
        """Drop cache entries so that exactly n positions remain."""

    def reset(self) -> None: ...


class HFBackend:
    def __init__(self, name: str, dtype: torch.dtype = torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).eval()
        gc = self.model.generation_config.eos_token_id
        ids = gc if isinstance(gc, list) else [gc if gc is not None else self.tokenizer.eos_token_id]
        self.eos_ids: set[int] = {int(i) for i in ids if i is not None}
        self.vocab_size = self.model.get_output_embeddings().weight.shape[0]
        self._cache: Any = None
        self._len = 0

    @property
    def length(self) -> int:
        return self._len

    @torch.inference_mode()
    def forward(self, tokens: list[int]) -> torch.Tensor:
        from transformers import DynamicCache

        if self._cache is None:
            self._cache = DynamicCache()
        ids = torch.tensor([tokens], dtype=torch.long)
        pos = torch.arange(self._len, self._len + len(tokens)).unsqueeze(0)
        out = self.model(
            input_ids=ids, position_ids=pos, past_key_values=self._cache, use_cache=True
        )
        self._len += len(tokens)
        return out.logits[0].float()

    def truncate(self, n: int) -> None:
        if self._cache is not None and n < self._len:
            self._cache.crop(n - self._len)  # negative = remove that many tokens
        self._len = min(n, self._len)

    def reset(self) -> None:
        self._cache = None
        self._len = 0


def load_backend(kind: str, name: str) -> ModelBackend:
    if kind == "hf":
        return HFBackend(name)  # type: ignore[return-value]
    if kind == "mlx":
        from specinfer.engine.mlx_backend import MLXBackend

        return MLXBackend(name)  # type: ignore[return-value]
    raise ValueError(f"unknown backend {kind!r}")
