# Architecture

Client -> FastAPI (`server/`) -> semantic cache (`cache/`, embedded Qdrant, cosine distance < 0.05,
temperature-0 requests only) -> speculative engine (`engine/`).

## Engine
- `ModelBackend` interface (`forward`, `truncate`, `reset`, `length`): `HFBackend` (PyTorch, used
  on the Linux dev box) and `MLXBackend` (Apple Silicon, untested here).
- Cache invariant between cycles: both KV caches hold `seq[:-1]`; the last committed token is the
  next cycle's pending input. A cycle: draft K tokens -> one target pass over `[last, d1..dK]` ->
  `rejection_sample` -> `KVCacheSynchronizer.commit` crops the target to `base+n+1`; the draft is
  cropped likewise, or fed `d_K` when all K were accepted.
- `sampler.rejection_sample` implements accept `min(1, p/q)`, residual resample
  `norm(max(0, p-q))`, and a bonus token from `p[K]` when all drafts are accepted. Draft and target
  use the same temperature/top-p transform, which is required for exactness. Temperature 0 is the
  one-hot special case (greedy), where speculative output equals baseline output.

## Speedup model
Expected tokens per cycle = (1 - a^(K+1)) / (1 - a); cost per cycle = K*c_D + c_T, so
S = (1 - a^(K+1)) / ((1 - a)(K*c + 1)) with c = c_D / c_T.
