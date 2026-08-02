"""Pure-PyTorch Mamba (selective SSM) for System B.

Fallback implementation used because mamba-ssm's fused CUDA kernels require a
CUDA toolkit (nvcc) that is not available in this environment. The selective
scan is computed with an exact log-depth (Hillis-Steele) inclusive scan over the
diagonal-recurrence segment semigroup, which is fully vectorized and runs on
either GPU or CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset


def _hillis_steel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One Hillis-Steele inclusive scan pass over the diagonal recurrence."""
    log_steps = int(math.ceil(math.log2(a.shape[1])))
    for d in range(log_steps):
        offset = 1 << d
        a_shift = torch.cat([torch.ones_like(a[:, :offset]), a[:, :-offset]], dim=1)
        b_shift = torch.cat([torch.zeros_like(b[:, :offset]), b[:, :-offset]], dim=1)
        a_new = a * a_shift
        b_new = a * b_shift + b
        a, b = a_new, b_new
    return b


def selective_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact parallel scan for h_t = a_t * h_{t-1} + b_t (elementwise).

    Custom autograd.Function: the backward computes the adjoint recurrence
    r_t = g_t + a_{t+1} * r_{t+1} with a reversed parallel scan, so memory
    stays at O(1) scan levels instead of O(log L) as with plain autograd.
    """
    return _SelectiveScanFn.apply(a, b)


class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return _hillis_steel(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        a, b = ctx.saved_tensors
        h = _hillis_steel(a, b)
        g = grad_output

        # Adjoint: r_t = g_t + a_{t+1} * r_{t+1} (scan from right to left).
        # Reversed recurrence R_j = G_j + A_j * R_{j-1} with R_j = r_{L-1-j},
        # G_j = g_{L-1-j}, A_j = a_{L-j} (so a_rev = flip(a) shifted by one).
        a_rev = torch.cat([torch.ones_like(a[:, :1]), torch.flip(a, dims=[1])[:, :-1]], dim=1)
        g_rev = torch.flip(g, dims=[1])
        r_rev = _hillis_steel(a_rev, g_rev)
        r = torch.flip(r_rev, dims=[1])

        grad_b = r
        h_shift = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        grad_a = r * h_shift
        return grad_a, grad_b


class SelectiveSSM(nn.Module):
    """One selective SSM block (conv + discretized diagonal SSM + gating)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        d_inner = int(expand * d_model)
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.checkpoint = checkpoint

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, bias=True, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        b_, l_, _ = x.shape

        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_.transpose(1, 2))[..., :l_]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        x_dbl = self.x_proj(x_conv)
        dt, b_proj, c_proj = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).unsqueeze(-1)  # (B, L, d_inner, 1)

        a = -torch.exp(self.A_log)  # (d_inner, d_state)
        a_bar = torch.exp(dt * a)  # (B, L, d_inner, d_state), in (0, 1]
        b_bar = b_proj.unsqueeze(2) * dt  # (B, L, 1, d_state)

        c_flat = a_bar.flatten(-2)  # (B, L, d_inner * d_state)
        b_flat = (b_bar * x_conv.unsqueeze(-1)).flatten(-2)

        h_flat = selective_scan(c_flat, b_flat)

        h = h_flat.unflatten(-1, (self.d_inner, self.d_state))
        y = (h * c_proj.unsqueeze(2)).sum(-1) + self.D * x_conv  # (B, L, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaSequenceClassifier(nn.Module):
    """Embedding -> Mamba blocks -> norm -> mean-pool -> per-disease head."""

    def __init__(
        self,
        vocab_size: int,
        n_diseases: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_diseases),
        )
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) int. Returns logits (B, n_diseases)."""
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        pooled = x.mean(dim=1)  # mean-pool over sequence length
        return self.head(pooled)
