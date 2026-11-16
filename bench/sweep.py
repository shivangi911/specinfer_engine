"""Sweep lookahead K and model precision on a few prompts (greedy).

python bench/sweep.py --dtypes float32 bfloat16 --ks 2 4 6
"""
from __future__ import annotations

import argparse
import json

import torch

from specinfer.config import settings
from specinfer.engine.model_loader import HFBackend
from specinfer.engine.speculative import GenStats, baseline_generate, speculative_generate

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function to merge two sorted lists into one sorted list.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
]


def run(gen) -> None:
    for _ in gen:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 6])
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default="docs/sweep_results.json")
    a = ap.parse_args()

    rows = []
    for dt in a.dtypes:
        dtype = getattr(torch, dt)
        draft, target = HFBackend(settings.draft_model, dtype), HFBackend(settings.target_model, dtype)
        prompts = [list(target.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            return_dict=True)["input_ids"]) for p in PROMPTS]
        b = GenStats()
        for pr in prompts:
            st = GenStats()
            run(baseline_generate(target, pr, target.eos_ids, a.max_new, stats=st))
            b.tokens += st.tokens; b.seconds += st.seconds
        rows.append({"dtype": dt, "k": 0, "tok_s": b.tok_per_s, "speedup": 1.0, "acceptance": None})
        print(rows[-1], flush=True)
        for k in a.ks:
            s = GenStats()
            for pr in prompts:
                st = GenStats()
                run(speculative_generate(draft, target, pr, target.eos_ids, a.max_new, k, stats=st))
                s.tokens += st.tokens; s.seconds += st.seconds
                s.drafted += st.drafted; s.accepted += st.accepted
            rows.append({"dtype": dt, "k": k, "tok_s": s.tok_per_s,
                         "speedup": s.tok_per_s / b.tok_per_s, "acceptance": s.acceptance_rate})
            print(rows[-1], flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
