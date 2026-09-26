"""Tests for F-05: focal / cost-sensitive objectives (polymas_ml.models.focal).

The gradient/Hessian closed forms are DERIVED; these tests pin them to the
reference loss via central finite differences so no recalled formula can
silently diverge from the definition.
"""

import numpy as np
import pytest

from polymas_ml.models.focal import (
    HESS_FLOOR,
    cost_sensitive_objective,
    focal_grad_hess,
    focal_loss_value,
    focal_objective,
)


Z = np.linspace(-6, 6, 601)


@pytest.mark.parametrize("y", [0, 1])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
def test_grad_matches_finite_differences(y, alpha, gamma) -> None:
    yv = np.full_like(Z, y, dtype=np.float64)
    g, _ = focal_grad_hess(yv, Z, alpha=alpha, gamma=gamma, clip_hess=False)
    h = 1e-6
    lp = focal_loss_value(yv, Z + h, alpha=alpha, gamma=gamma)
    lm = focal_loss_value(yv, Z - h, alpha=alpha, gamma=gamma)
    numeric = (lp - lm) / (2 * h)
    np.testing.assert_allclose(g, numeric, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("y", [0, 1])
@pytest.mark.parametrize("alpha", [0.25, 0.5])
@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
def test_hess_matches_finite_differences(y, alpha, gamma) -> None:
    yv = np.full_like(Z, y, dtype=np.float64)
    g_plus, _ = focal_grad_hess(yv, Z + 1e-5, alpha=alpha, gamma=gamma, clip_hess=False)
    g_minus, _ = focal_grad_hess(yv, Z - 1e-5, alpha=alpha, gamma=gamma, clip_hess=False)
    numeric = (g_plus - g_minus) / 2e-5
    _, h = focal_grad_hess(yv, Z, alpha=alpha, gamma=gamma, clip_hess=False)
    # Assert only where the exact Hessian is positive (the exact one can go
    # negative at extreme scores for gamma > 0 — documented property).
    mask = h > 1e-4
    np.testing.assert_allclose(h[mask], numeric[mask], atol=1e-4, rtol=1e-3)


def test_hess_matches_gradient_derivative() -> None:
    """Ground-truth Hessian check: the gradient is already pinned to the
    loss by finite differences, so differentiating THAT gradient
    numerically gives an authoritative second derivative. (A 5-point loss
    stencil is ill-conditioned near z->-inf where the loss spans huge
    dynamic range; this check avoids that cancellation.)"""
    for yv, alpha, gamma in [(1.0, 0.25, 2.0), (1.0, 0.75, 1.0),
                             (0.0, 0.25, 2.0), (0.0, 0.9, 1.5)]:
        yy = np.full_like(Z, yv)
        h = 1e-5
        gp, _ = focal_grad_hess(yy, Z + h, alpha=alpha, gamma=gamma, clip_hess=False)
        gm, _ = focal_grad_hess(yy, Z - h, alpha=alpha, gamma=gamma, clip_hess=False)
        numeric = (gp - gm) / (2 * h)
        _, hess = focal_grad_hess(yy, Z, alpha=alpha, gamma=gamma, clip_hess=False)
        mask = (Z > -5) & (Z < 5)
        np.testing.assert_allclose(hess[mask], numeric[mask], atol=1e-7, rtol=1e-6)


def test_hess_clip_enforced() -> None:
    """The training objective never returns a Hessian below HESS_FLOOR.
    The exact gamma=2 Hessian is negative for y=1 at small p (documented
    in the module docstring and verified by the stencil test above); the
    floor is what makes the objective usable by LightGBM."""
    yv = np.ones_like(Z)
    _, h_exact = focal_grad_hess(yv, Z, gamma=2.0, clip_hess=False)
    assert (h_exact < 0).any()          # the negative region is real
    _, h_clipped = focal_grad_hess(yv, Z, gamma=2.0, clip_hess=True)
    assert (h_clipped >= HESS_FLOOR - 1e-15).all()


def test_gamma_zero_reduces_to_weighted_bce() -> None:
    yv = (Z > 0).astype(np.float64)
    g, h = focal_grad_hess(yv, Z, alpha=0.3, gamma=0.0, clip_hess=False)
    p = 1.0 / (1.0 + np.exp(-Z))
    w = np.where(yv == 1, 0.3, 0.7)
    np.testing.assert_allclose(g, w * (p - yv), atol=1e-10)
    np.testing.assert_allclose(h, w * p * (1.0 - p), atol=1e-10)


def test_objective_factory_signature() -> None:
    yv = (Z > 0).astype(np.float64)
    obj = focal_objective(alpha=0.25, gamma=2.0)
    g, h = obj(yv, Z)
    g2, h2 = focal_grad_hess(yv, Z)
    np.testing.assert_allclose(g, g2)
    np.testing.assert_allclose(h, h2)


def test_cost_sensitive_matches_weighted_bce() -> None:
    yv = (Z > 0).astype(np.float64)
    spw = 3.7
    obj = cost_sensitive_objective(spw)
    g, h = obj(yv, Z)
    p = 1.0 / (1.0 + np.exp(-Z))
    w = np.where(yv == 1, spw, 1.0)
    np.testing.assert_allclose(g, w * (p - yv), atol=1e-10)
    np.testing.assert_allclose(h, w * p * (1.0 - p), atol=1e-10)


def test_lightgbm_accepts_objective() -> None:
    """End-to-end: LightGBM trains with the custom focal objective.
    With a custom objective, predict_proba returns the RAW margin (1-D);
    the caller applies the link (sigmoid) manually — the F-05 runner does
    the same, so the test pins that contract."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=30, objective=focal_objective())
    model.fit(X, y)
    raw = np.asarray(model.predict_proba(X)).squeeze()
    assert raw.ndim == 1
    proba = 1.0 / (1.0 + np.exp(-raw))
    assert ((proba > 0) & (proba < 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, proba) > 0.6
