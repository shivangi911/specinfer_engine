"""CLI: python -m specinfer.engine.sampler --mode baseline|speculative ..."""
from __future__ import annotations

import argparse

from specinfer.config import settings
from specinfer.engine.model_loader import HFBackend
from specinfer.engine.speculative import GenStats, baseline_generate, speculative_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "speculative"], default="speculative")
    ap.add_argument("--model", default=settings.target_model, help="baseline target")
    ap.add_argument("--draft", default=settings.draft_model)
    ap.add_argument("--target", default=settings.target_model)
    ap.add_argument("--prompt", default="Write a Python function that checks if a number is prime.")
    ap.add_argument("--k", type=int, default=settings.lookahead_k)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    target = HFBackend(a.model if a.mode == "baseline" else a.target)
    msgs = [{"role": "user", "content": a.prompt}]
    ids = target.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True)
    prompt = list(ids["input_ids"])  # type: ignore[index,call-overload]
    st = GenStats()
    if a.mode == "baseline":
        gen = baseline_generate(target, prompt, target.eos_ids, a.max_new, a.temperature, stats=st)
    else:
        draft = HFBackend(a.draft)
        gen = speculative_generate(
            draft, target, prompt, target.eos_ids, a.max_new, a.k, a.temperature, stats=st
        )
    for _ in gen:
        pass
    print(target.tokenizer.decode(st.token_ids, skip_special_tokens=True))
    print(f"\n[{a.mode}] tokens={st.tokens} tok/s={st.tok_per_s:.2f} "
          f"acceptance={st.acceptance_rate:.2%} cycles={st.cycles}")


if __name__ == "__main__":
    main()
