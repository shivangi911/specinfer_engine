import torch

from specinfer.engine.sampler import logits_to_probs, rejection_sample


def _dist(v, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(v, generator=g) * 2, -1)


def test_first_token_matches_target_distribution():
    """Chi-square style check: emitted first token ~ p[0] even though drafted from q."""
    v, trials = 6, 40000
    p0, q0 = _dist(v, 1), _dist(v, 2)
    p = torch.stack([p0, _dist(v, 3)])
    q = q0.unsqueeze(0)
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(v)
    for _ in range(trials):
        tok = torch.multinomial(q0, 1, generator=g)
        n, nxt = rejection_sample(tok, q, p, g)
        first = int(tok[0]) if n >= 1 else nxt
        counts[first] += 1
    emp = counts / trials
    chi2 = (trials * (emp - p0) ** 2 / p0).sum().item()
    assert chi2 < 20.5, (chi2, emp, p0)  # df=5, p~0.001


def test_all_accept_when_q_equals_p():
    v, k = 8, 4
    p = torch.stack([_dist(v, i) for i in range(k + 1)])
    q = p[:k].clone()
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        toks = torch.stack([torch.multinomial(q[i], 1, generator=g)[0] for i in range(k)])
        n, _ = rejection_sample(toks, q, p, g)
        assert n == k


def test_all_reject_disjoint_support():
    q = torch.tensor([[1.0, 0, 0, 0]] * 3)
    p = torch.tensor([[0, 1.0, 0, 0]] * 4)
    n, nxt = rejection_sample(torch.tensor([0, 0, 0]), q, p)
    assert (n, nxt) == (0, 1)


def test_greedy_probs_is_one_hot_and_top_p():
    lg = torch.tensor([[1.0, 3.0, 2.0]])
    assert logits_to_probs(lg, 0.0).tolist() == [[0, 1, 0]]
    pr = logits_to_probs(lg, 1.0, top_p=0.5)
    assert abs(pr.sum().item() - 1) < 1e-6 and pr[0, 0] == 0
