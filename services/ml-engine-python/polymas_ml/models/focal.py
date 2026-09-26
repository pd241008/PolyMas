"""F-05: class-imbalance-aware objectives (ROADMAP Phase 3).

Focal loss (Lin et al. 2017) and cost-sensitive weighting as LightGBM
custom objectives, for the paired ablation in scripts/f05_focal_loss.py:

    control  : standard binary logloss
    focal    : alpha-weighted focal loss, gamma=2, alpha=0.25 (paper
               default — pre-registered, NOT tuned)
    cost-sens: logloss with positives up-weighted (scale_pos_weight)

Definition (paper notation, per-sample class weighting):

    L(y, z) = -alpha_t * (1 - p_t)^gamma * log(p_t),
    p = sigmoid(z),  p_t = p if y == 1 else 1 - p,
    alpha_t = alpha if y == 1 else 1 - alpha.

Gradient/Hessian: rather than error-prone expanded closed forms, the
second derivative is composed from its chain-rule factors, each of which
is a simple one-line expression verified by finite differences in
tests/test_focal.py:

    dL/dz   = F'(p) * p(1-p)                    [p_t branch-dependent]
    d2L/dz2 = F''(p) * (p(1-p))^2 + F'(p) * p(1-p)(1-2p)

with, for the y=1 branch  F1(p) = -alpha (1-p)^g log p:
    F1'(p)  =  alpha*g*(1-p)^(g-1)*log p - alpha*(1-p)^g/p
    F1''(p) = -alpha*g*(g-1)*(1-p)^(g-2)*log p
              + 2*alpha*g*(1-p)^(g-1)/p + alpha*(1-p)^g/p^2
and the y=0 branch as F0(p0) = -(1-alpha) p^g log(p0), p0 = 1-p, with
dp0/dz = -p(1-p):
    F0'(p0)  = -(1-alpha)*g*p^(g-1)*log(p0) + (1-alpha)*p^g/p0
    F0''(p0) = -(1-alpha)*g*(g-1)*p^(g-2)*log(p0)
               + 2*(1-alpha)*g*p^(g-1)/p0 + (1-alpha)*p^g/p0^2
(the composed forms reduce to weighted-BCE grad/hess at gamma=0 and match
the classic simplified closed forms at gamma=1 — both asserted in tests).

The exact focal Hessian is NEGATIVE on a small-p region of the y=1
branch (e.g. gamma=2, alpha=0.25, p~0.01 gives h ~ -1e-2; verified
against a 5-point stencil of the loss in tests), a known property of the
focal loss. The training objective clips at HESS_FLOOR (LightGBM
requires hess > 0) — a visible, tested floor, not a silent corrector.
"""
from __future__ import annotations

import numpy as np

HESS_FLOOR = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def focal_loss_value(
    y_true: np.ndarray, z: np.ndarray, alpha: float = 0.25, gamma: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-sample focal loss given raw scores z (reference implementation)."""
    p = _sigmoid(z)
    alpha_t = np.where(np.asarray(y_true) == 1, alpha, 1.0 - alpha)
    p_t = np.where(np.asarray(y_true) == 1, p, 1.0 - p)
    return -alpha_t * (1.0 - p_t) ** gamma * np.log(np.clip(p_t, eps, 1.0))


def focal_grad_hess(
    y_true: np.ndarray, z: np.ndarray, alpha: float = 0.25, gamma: float = 2.0,
    eps: float = 1e-12, clip_hess: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of the focal loss w.r.t. raw scores z.

    clip_hess=False exposes the un-clipped second derivative (testing)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = _sigmoid(z)
    pc = np.clip(p, eps, 1.0 - eps)
    u = 1.0 - pc                      # 1 - p (clipped)
    g = float(gamma)
    a = float(alpha)
    log_p = np.log(pc)
    log_u = np.log(u)
    pu = pc * u                       # dp/dz magnitude

    # y = 1 branch: F1(p) = -a * u^g * log p   (differentiate wrt p)
    F1p = a * g * u ** (g - 1) * log_p - a * u ** g / pc
    F1pp = (-a * g * (g - 1) * u ** (g - 2) * log_p
            + 2 * a * g * u ** (g - 1) / pc
            + a * u ** g / pc ** 2)
    grad_pos = F1p * pu
    hess_pos = F1pp * pu ** 2 + F1p * pu * (u - pc)

    # y = 0 branch: F0(p0) = -(1-a) * p^g * log(p0), p0 = u  (wrt p0).
    # d/dp0[p^g] = -g*p^(g-1) (p = 1 - p0), so the signs differ from F1p.
    F0p = (1 - a) * g * pc ** (g - 1) * log_u - (1 - a) * pc ** g / u
    F0pp = (-(1 - a) * g * (g - 1) * pc ** (g - 2) * log_u
            + 2 * (1 - a) * g * pc ** (g - 1) / u
            + (1 - a) * pc ** g / u ** 2)
    # dp0/dz = -pu;  d2p0/dz2 = -pu(u-p)
    grad_neg = F0p * (-pu)
    hess_neg = F0pp * pu ** 2 + F0p * (-pu * (u - pc))

    grad = np.where(y == 1, grad_pos, grad_neg)
    hess = np.where(y == 1, hess_pos, hess_neg)
    if clip_hess:
        hess = np.maximum(hess, HESS_FLOOR)
    return grad, hess


def focal_objective(alpha: float = 0.25, gamma: float = 2.0):
    """LightGBM sklearn-API custom objective: callable(y_true, y_pred_raw)
    -> (grad, hess)."""
    def objective(y_true: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return focal_grad_hess(y_true, z, alpha=alpha, gamma=gamma)
    return objective


def cost_sensitive_objective(scale_pos_weight: float):
    """Logloss objective with positives up-weighted by scale_pos_weight,
    expressed through the same custom-objective interface so the F-05
    ablation differs ONLY in the loss."""
    def objective(y_true: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_true, dtype=np.float64)
        p = _sigmoid(z)
        w = np.where(y == 1, scale_pos_weight, 1.0)
        grad = w * (p - y)
        hess = np.maximum(w * p * (1.0 - p), HESS_FLOOR)
        return grad, hess
    return objective
