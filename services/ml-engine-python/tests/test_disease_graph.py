"""Tests for F-02: MAS-aware disease-graph head (polymas_ml.models.disease_graph)."""

import numpy as np
import torch

from polymas_ml.data.patients import MAS_EXCLUSIONS, MAS_PAIRWISE_ODDS
from polymas_ml.models.disease_graph import (
    DISEASE_LABELS,
    DiseaseGraphHead,
    IndependentHeadControl,
    mas_adjacency_init,
    predict_head,
    train_head,
)


def test_adjacency_init_matches_published_table() -> None:
    W = mas_adjacency_init(scale=0.5)
    idx = {d: i for i, d in enumerate(DISEASE_LABELS)}
    assert torch.diagonal(W).abs().sum() == 0
    assert torch.allclose(W, W.T)
    for (a, b), lor in MAS_PAIRWISE_ODDS.items():
        assert abs(W[idx[a], idx[b]].item() - 0.5 * lor) < 1e-6
    for a, b in MAS_EXCLUSIONS:
        assert W[idx[a], idx[b]] < 0


def test_exclusion_pairs_stay_negative_after_training_step() -> None:
    model = DiseaseGraphHead()
    # Force the raw params of exclusion pairs positive; the effective
    # adjacency must still be negative (softplus re-parameterization).
    with torch.no_grad():
        raw = model.raw_W
        raw[model.excl_mask] = 5.0
    W = model.effective_W()
    assert (W[model.excl_mask] < 0).all()
    assert torch.diagonal(W).abs().sum() == 0


def test_graph_step_uses_adjacency() -> None:
    torch.manual_seed(0)
    model = DiseaseGraphHead()
    p = torch.rand(4, 7)
    # With raw weights zeroed, free entries contribute 0; exclusion entries
    # sit at -softplus(0) = -0.693 by design (they can never turn positive).
    with torch.no_grad():
        model.raw_W.zero_()
    msg0 = model.graph_step(p)
    W0 = model.effective_W()
    assert torch.allclose(msg0, torch.tanh(p @ W0.T), atol=1e-6)
    # Free off-diagonals are exactly zero with zeroed raw weights.
    free = ~model.excl_mask
    assert W0[free].abs().sum() == 0
    # Restoring the MAS init makes the message stronger (message passing on).
    with torch.no_grad():
        model.raw_W.copy_(mas_adjacency_init())
    msg1 = model.graph_step(p)
    assert msg1.abs().max() > msg0.abs().max()


def _synthetic_task(n: int = 3000, seed: int = 0):
    """Base probabilities whose noise carries a real MAS co-occurrence signal
    in SLE/T1D so the graph path has something genuine to exploit."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    p = rng.uniform(0.05, 0.6, size=(n, 7))
    # Co-occurrence: high latent raises SLE and T1D together.
    p[:, 1] = 1 / (1 + np.exp(-(3 * latent - 1)))
    p[:, 4] = 1 / (1 + np.exp(-(2 * latent + 0.5)))
    Y = (rng.uniform(size=(n, 7)) < p).astype(np.float32)
    # Labels share the latent, so p_base alone is informative; the head's job
    # is to combine the correlated channels without hurting AUROC.
    return p.astype(np.float32), Y


def test_train_head_learns_and_beats_chance() -> None:
    X, Y = _synthetic_task()
    n = len(X)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train_idx, val_idx = perm[:2400], perm[2400:]
    head = DiseaseGraphHead()
    hist = train_head(head, X, Y, train_idx, val_idx, epochs=8, device="cpu")
    out = predict_head(head, X[val_idx], device="cpu")
    from sklearn.metrics import roc_auc_score

    aucs = [roc_auc_score(Y[val_idx, k], out[:, k])
            for k in range(7) if len(np.unique(Y[val_idx, k])) > 1]
    assert np.mean(aucs) > 0.55
    assert hist["best_val_macro_auroc"] > 0.55


def test_control_and_head_share_interface() -> None:
    X, Y = _synthetic_task(n=600, seed=1)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    train_idx, val_idx = perm[:450], perm[450:]
    for model in (DiseaseGraphHead(), IndependentHeadControl()):
        train_head(model, X, Y, train_idx, val_idx, epochs=3, device="cpu")
        out = predict_head(model, X[val_idx], device="cpu")
        assert out.shape == (150, 7)
        assert ((out > 0) & (out < 1)).all()


def test_adjacency_report_lists_all_published_pairs() -> None:
    model = DiseaseGraphHead()
    rep = model.adjacency_report()
    n_expected = 2 * len(MAS_PAIRWISE_ODDS) + 2 * len(MAS_EXCLUSIONS)
    assert len(rep) == n_expected
    for entry in rep.values():
        assert "init" in entry and "learned" in entry
