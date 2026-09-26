"""F-01: LD-aware GNN over the locus graph (System C, ROADMAP Phase 3).

Pre-registered claim: a message-passing GNN whose edges are REAL LD-r²
between panel loci matches or beats the GBM ensemble on held-out AUROC.
Tolerance: within ±0.02 macro AUROC of System A counts as "matches";
strictly above = "beats".

Design (pre-registered before the run):
  - Nodes: the 14 patient-typed panel loci (each patient's genotype
    dosage + haplotype/epistasis feature values live on the nodes).
  - Edges: pairwise LD r² computed from the 91-locus x 2504-donor
    substrate dosage matrix (real 1000G haplotypes), thresholded at
    r² >= 0.2; the 14 typed nodes inherit edges to ALL substrate loci,
    so information flows typed-node -> tag-node -> typed-node across 2
    rounds. Edge weights = r² (R2 provenance: the full r² matrix is
    written alongside the model).
  - LD-edge audit (pre-registered): every claimed HLA pair in the panel
    documentation must have r² >= 0.3 in the substrate, and the strongest
    learned edge usage must sit inside the MHC block — reported, not
    tuned.
  - Model: 2 rounds of (mean-aggregate -> linear -> ReLU) message
    passing, LayerNorm, mean-pool over nodes, linear per-disease head.
    Capacity deliberately modest; the claim is about the EDGE STRUCTURE.
  - Split: canonical F-12 train/val/test on patients; loss
    BCEWithLogits over the 7 diseases.

Evidence: <results_root>/f01_ld_gnn/
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LDGNN(nn.Module):
    """2-round message-passing GNN over the locus graph."""

    def __init__(
        self,
        n_node_features: int,
        n_diseases: int = 7,
        hidden: int = 64,
        n_rounds: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_rounds = n_rounds
        self.in_proj = nn.Linear(n_node_features, hidden)
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_rounds)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_rounds)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_diseases))

    def forward(
        self,
        x: torch.Tensor,        # (B, n_nodes, n_node_features)
        adj: torch.Tensor,      # (n_nodes, n_nodes), row-normalized, zero diag
    ) -> torch.Tensor:
        h = torch.relu(self.in_proj(x))
        for i in range(self.n_rounds):
            agg = torch.matmul(adj, h)          # mean-aggregate over neighbors
            h = self.norms[i](h + self.drop(self.msg[i](agg)))   # residual MP
        pooled = h.mean(dim=1)                  # mean-pool over nodes
        return self.head(pooled)


def row_normalize_adj(adj_raw: torch.Tensor) -> torch.Tensor:
    """Row-normalize an adjacency (self-loops excluded); rows with no
    neighbors fall back to self-weight 1 (isolated node keeps its features)."""
    deg = adj_raw.sum(dim=1, keepdim=True)
    safe = torch.where(deg > 0, deg, torch.ones_like(deg))
    out = adj_raw / safe
    isolated = (deg.squeeze(1) == 0)
    if isolated.any():
        eye = torch.eye(adj_raw.shape[0], device=adj_raw.device, dtype=adj_raw.dtype)
        out = torch.where(isolated.unsqueeze(1), eye, out)
    return out


def build_ld_audit(ld_matrix: pd_like) -> dict:
    """R2 provenance for the LD edges: strongest pairs, MHC block check.
    (pd_like: pandas DataFrame of r2 values.)"""
    import pandas as pd
    df = ld_matrix.copy()
    values = df.to_numpy().copy()   # .values can be a read-only view
    np.fill_diagonal(values, 0.0)
    df = pd.DataFrame(values, index=df.index, columns=df.columns)
    pairs = (
        df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        .stack().sort_values(ascending=False)
    )
    top = [{"a": a, "b": b, "r2": round(float(v), 4)} for (a, b), v in pairs.head(12).items()]
    mhc = df.loc[[i for i in df.index if i.startswith("rs927") or i.startswith("rs2187") or i.startswith("rs3135")],
                 [j for j in df.columns if j.startswith("rs927") or j.startswith("rs2187") or j.startswith("rs3135")]]
    return {
        "n_edges_ge_0.2": int((pairs >= 0.2).sum()),
        "n_edges_ge_0.5": int((pairs >= 0.5).sum()),
        "top_pairs": top,
        "mhc_block_max_r2": round(float(mhc.values.max()), 4) if mhc.size else None,
    }
