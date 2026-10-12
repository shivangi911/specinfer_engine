"""HumanEval benchmark: baseline vs speculative (greedy) with pass@1 check.

python bench/humaneval.py --n 10 --max-new 256 --k 4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile

from datasets import load_dataset

from specinfer.config import settings
from specinfer.engine.model_loader import HFBackend
from specinfer.engine.speculative import GenStats, baseline_generate, speculative_generate


def extract_code(text: str, prompt: str, entry: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL) or re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)
    code = m.group(1) if m else text
    return code if f"def {entry}" in code else prompt + code


def run_tests(program: str, test: str, entry: str, timeout: int = 20) -> bool:
    src = f"{program}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, check=False,
                              timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def theoretical_speedup(alpha: float, k: int, c: float) -> float:
    """c = draft cost / target cost per token. Expected tokens/cycle over cost/cycle."""
    exp = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1
    return exp / (c * k + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--k", type=int, default=settings.lookahead_k)
    ap.add_argument("--out", default="docs/bench_results.json")
    a = ap.parse_args()

    ds = load_dataset("openai/openai_humaneval", split="test").select(range(a.n))
    draft, target = HFBackend(settings.draft_model), HFBackend(settings.target_model)
    res = {"baseline": GenStats(), "speculative": GenStats()}
    passed = {"baseline": 0, "speculative": 0}
    same = 0
    for i, ex in enumerate(ds):
        msgs = [{"role": "user", "content":
                 "Complete this Python function. Return the full function in a code block.\n\n"
                 f"```python\n{ex['prompt']}```"}]
        prompt = list(target.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_dict=True)["input_ids"])
        outs = {}
        for mode in ("baseline", "speculative"):
            st = GenStats()
            if mode == "baseline":
                g = baseline_generate(target, prompt, target.eos_ids, a.max_new, stats=st)
            else:
                g = speculative_generate(draft, target, prompt, target.eos_ids, a.max_new, a.k,
                                         stats=st)
            for _ in g:
                pass
            text = target.tokenizer.decode(st.token_ids, skip_special_tokens=True)
            outs[mode] = text
            agg = res[mode]
            agg.tokens += st.tokens; agg.seconds += st.seconds
            agg.drafted += st.drafted; agg.accepted += st.accepted; agg.cycles += st.cycles
            code = extract_code(text, ex["prompt"], ex["entry_point"])
            passed[mode] += run_tests(code, ex["test"], ex["entry_point"])
        same += outs["baseline"] == outs["speculative"]
        print(f"[{i + 1}/{a.n}] pass b/s = {passed['baseline']}/{passed['speculative']} "
              f"identical={same}", flush=True)

    b, s = res["baseline"], res["speculative"]
    alpha = s.acceptance_rate
    out = {
        "problems": a.n, "k": a.k, "max_new": a.max_new,
        "draft": settings.draft_model, "target": settings.target_model,
        "baseline_tok_s": b.tok_per_s, "spec_tok_s": s.tok_per_s,
        "speedup": s.tok_per_s / b.tok_per_s if b.tok_per_s else 0,
        "acceptance_rate": alpha,
        "theoretical_speedup_at_c=0.33": theoretical_speedup(alpha, a.k, 0.33),
        "pass1_baseline": passed["baseline"] / a.n, "pass1_speculative": passed["speculative"] / a.n,
        "identical_outputs": f"{same}/{a.n}",
    }
    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
