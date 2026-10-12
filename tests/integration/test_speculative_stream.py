"""Needs the Qwen models in the local HF cache; run with `pytest -m slow`."""
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def models():
    from specinfer.config import settings
    from specinfer.engine.model_loader import HFBackend

    return HFBackend(settings.draft_model), HFBackend(settings.target_model)


def test_greedy_speculative_matches_baseline(models):
    from specinfer.engine.speculative import baseline_generate, speculative_generate

    draft, target = models
    msgs = [{"role": "user", "content": "Write a python function to reverse a string."}]
    prompt = list(
        target.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True)[
            "input_ids"
        ]
    )
    base = [t for c in baseline_generate(target, prompt, target.eos_ids, 24) for t in c]
    spec = [t for c in speculative_generate(draft, target, prompt, target.eos_ids, 24, 4) for t in c]
    # fp32 CPU: batched verification can differ from 1-token decode by float noise; require
    # near-identical output (exact prefix of at least 90%).
    n = min(len(base), len(spec))
    common = next((i for i in range(n) if base[i] != spec[i]), n)
    assert common >= 0.9 * n
