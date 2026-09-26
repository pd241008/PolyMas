"""F-02: MAS-aware disease-graph head (ROADMAP Phase 3).

A multi-label head over base-learner probabilities with a LEARNED
disease-disease adjacency matrix initialized from the published MAS odds
table (MAS_PAIRWISE_ODDS log-ORs; incoercible MAS_EXCLUSIONS pairs start
negative and are structurally kept negative). Claim under test: joint
modeling of the label graph improves co-occurrence calibration vs the
independent-head ensemble, at no AUROC cost.

Pre-registered F-02 tolerance (ROADMAP): >= neutral on AUROC (−0.01 floor)
AND improved pairwise-phi recovery. This module ships the model; the
paired evaluation (same split, same base learners) lives in
scripts/f02_disease_graph.py.

Architecture (per patient):
    p_base (7 logits) -> concat[ p_base, graph_step(p_base) ] -> MLP -> 7 outputs
where graph_step is one round of message passing:
    h_i = sigma( b_i + sum_{j != i} W_ij * p_j ),  W learned (7x7, zero
    diagonal), initialized from the MAS odds table; exclusion pairs are
    re-parameterized as W_ij = -softplus(raw) so they can never turn
    positive during training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from polymas_ml.data.patients import MAS_EXCLUSIONS, MAS_PAIRWISE_ODDS

DISEASE_LABELS = ["RA", "SLE", "SJOGRENS", "AITD", "T1D", "VITILIGO", "MS"]


def mas_adjacency_init(
    labels: list[str] = DISEASE_LABELS,
    scale: float = 0.5,
) -> torch.Tensor:
    """7x7 weight matrix from the published MAS log-OR affinities.

    Off-diagonal entries: scale * log(OR) for embedded pairs, -scale for
    incoercible exclusion pairs (antagonistic), 0 for unlisted pairs
    (learnable but started at no opinion). Diagonal zero — self terms are
    the MLP's own job.
    """
    n = len(labels)
    W = torch.zeros(n, n)
    idx = {d: i for i, d in enumerate(labels)}
    for (a, b), lor in MAS_PAIRWISE_ODDS.items():
        if a in idx and b in idx:
            W[idx[a], idx[b]] = scale * lor
            W[idx[b], idx[a]] = scale * lor
    for a, b in MAS_EXCLUSIONS:
        if a in idx and b in idx:
            W[idx[a], idx[b]] = -scale
            W[idx[b], idx[a]] = -scale
    return W


def _exclusion_mask(labels: list[str] = DISEASE_LABELS) -> torch.Tensor:
    """Boolean mask marking the off-diagonal entries that must stay <= 0."""
    n = len(labels)
    mask = torch.zeros(n, n, dtype=torch.bool)
    idx = {d: i for i, d in enumerate(labels)}
    for a, b in MAS_EXCLUSIONS:
        mask[idx[a], idx[b]] = True
        mask[idx[b], idx[a]] = True
    return mask


class DiseaseGraphHead(nn.Module):
    """MLP over [base_probs, graph_message] with learned MAS-init adjacency."""

    def __init__(
        self,
        n_diseases: int = 7,
        hidden: int = 64,
        adjacency_scale: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_diseases = n_diseases
        W0 = mas_adjacency_init(DISEASE_LABELS[:n_diseases], adjacency_scale)
        self.raw_W = nn.Parameter(W0)
        self.register_buffer("excl_mask", _exclusion_mask(DISEASE_LABELS[:n_diseases]))

        self.mlp = nn.Sequential(
            nn.Linear(2 * n_diseases, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_diseases),
        )

    def effective_W(self) -> torch.Tensor:
        """Adjacency actually used: free params elsewhere, exclusion pairs
        re-parameterized as -softplus(raw) (always negative), diagonal zero."""
        W = self.raw_W.clone()
        neg = -nn.functional.softplus(W[self.excl_mask])
        W = W.masked_scatter(self.excl_mask, neg)
        n = self.n_diseases
        return W - torch.diag(W.diagonal())

    def graph_step(self, p: torch.Tensor) -> torch.Tensor:
        """One message-passing round: h_i = tanh(b_i + sum_j W_ij p_j)."""
        W = self.effective_W()
        b = torch.zeros(self.n_diseases, device=p.device, dtype=p.dtype)
        return torch.tanh(p @ W.T + b)

    def forward(self, p_base: torch.Tensor) -> torch.Tensor:
        msg = self.graph_step(p_base)
        z = self.mlp(torch.cat([p_base, msg], dim=-1))
        return z

    def adjacency_report(self) -> dict:
        """R2 provenance: learned vs published adjacency."""
        W = self.effective_W().detach().cpu().numpy()
        W0 = mas_adjacency_init(DISEASE_LABELS[: self.n_diseases]).numpy()
        labels = DISEASE_LABELS[: self.n_diseases]
        published = {}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i != j and W0[i, j] != 0:
                    published[f"{a}|{b}"] = {
                        "init": float(W0[i, j]),
                        "learned": round(float(W[i, j]), 4),
                    }
        return published


class IndependentHeadControl(nn.Module):
    """Matched-capacity MLP with NO graph path (7 -> hidden -> 7), the honest
    control for the phi-recovery comparison: same depth/width on the same
    inputs, minus the disease-disease messages."""

    def __init__(
        self,
        n_diseases: int = 7,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_diseases = n_diseases
        self.mlp = nn.Sequential(
            nn.Linear(n_diseases, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_diseases),
        )

    def forward(self, p_base: torch.Tensor) -> torch.Tensor:
        return self.mlp(p_base)


def train_head(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Train either head with BCEWithLogitsLoss + AdamW; early-stop on mean
    val AUROC macro (computed with sklearn). Returns training history."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    va = torch.tensor(val_idx, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state, history = -np.inf, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        for s in range(0, len(perm), batch_size):
            b = torch.tensor(perm[s : s + batch_size], dtype=torch.long, device=device)
            opt.zero_grad()
            out = model(Xt[b])
            loss = loss_fn(out, Yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_out = model(Xt[va]).sigmoid().cpu().numpy()
        aucs = []
        va_np = np.asarray(val_idx)
        for k in range(Y.shape[1]):
            yk = Y[va_np, k]
            if len(np.unique(yk)) > 1:
                aucs.append(roc_auc_score(yk, val_out[:, k]))
        macro = float(np.mean(aucs)) if aucs else float("nan")
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_macro_auroc": macro})
        if macro > best_val:
            best_val, best_state = macro, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_macro_auroc": best_val, "history": history}


def predict_head(model: nn.Module, X: np.ndarray, device: str | None = None) -> np.ndarray:
    """Sigmoid outputs (n, 7) on CPU."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=dev)).sigmoid().cpu().numpy()
    model.train(was_training)
    return out
