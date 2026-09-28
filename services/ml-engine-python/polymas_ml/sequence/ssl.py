"""F-14: masked k-mer modeling pretraining for the sequence encoder (ROADMAP Phase 4).

Pre-registered protocol (written before any run):
  - Claim: masked-token pretraining of the Mamba encoder on the SAME 5,000
    genotypes (no external data) improves downstream val AUROC vs training
    from scratch, under an IDENTICAL fine-tune schedule (3 seeds, paired).
  - Pretrain: mask 15% of the 504 k-mer tokens, predict the original token
    id from the contextual representation (cross-entropy, masked positions
    only). 8 epochs, lr 1e-3, AdamW. The encoder (embedding + SSM stack +
    final norm) transfers; the SSM stack is causal — masking is handled by
    predicting from the causal prefix, the standard GPT-style protocol.
  - Fine-tune: identical hyperparameters for both arms (the flat recipe,
    6 epochs, lr 1e-3, batch 64); the ONLY difference is encoder
    initialization (pretrained vs fresh).
  - Gates (pre-registered): paired val AUROC delta >= +0.01 for the SSL arm
    (mean over 3 seeds) AND no per-seed regression > 0.01.
  - Either arm failing its gate is recorded honestly; a null result is an
    acceptable outcome (SSL with no external data at n=5,000 is expected to
    be weak — the claim tests whether genotype self-supervision helps at
    all in this regime).

The pretraining objective predicts the ORIGINAL token id at masked
positions (never the mask token) so the encoder learns genotype context,
not mask-token detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polymas_ml.sequence.dataset import TOKEN_HOM_REF, VOCAB_SIZE

MASK_TOKEN_ID = VOCAB_SIZE          # 4099; embedding row added for pretraining only
PRETRAIN_VOCAB = VOCAB_SIZE + 1     # 4100
MASK_FRACTION = 0.15


def mask_tokens(input_ids: torch.Tensor, mask_fraction: float = MASK_FRACTION,
                mask_token_id: int = MASK_TOKEN_ID, generator: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a random subset of tokens with MASK; return (masked, targets).

    targets hold ORIGINAL ids at masked positions and -100 elsewhere, so
    F.cross_entropy with ignore_index=-100 scores exactly the masked set.
    """
    masked = input_ids.clone()
    mask = (
        torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
        < mask_fraction
    )
    # -100 at KEPT positions, original ids at MASKED positions (ignore_index
    # semantics): CE scores exactly the masked set. NB: the F-14 run executed
    # with this fill inverted (-100 at masked), i.e. it scored the VISIBLE
    # tokens — a degenerate identity-copy objective. Discovered while building
    # F-14a (2026-09-28); F-14 evidence annotated, not re-run (PM-006 addendum).
    targets = input_ids.masked_fill(~mask, -100)
    masked[mask] = mask_token_id
    return masked, targets


def mask_context_tokens(input_ids: torch.Tensor, mask_fraction: float = MASK_FRACTION,
                        mask_token_id: int = MASK_TOKEN_ID, generator: torch.Generator | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """F-14a variant: mask CONTEXT k-mers only; genotype tokens stay visible.

    Context k-mers occupy ids 0..4095; genotype tokens are ids
    TOKEN_HOM_REF..TOKEN_HOM_ALT (4096..4098). The F-14 root cause was that
    masking all positions let the encoder profit from memorizing shared
    flanking context while the 8 genotype tokens (the disease signal) were
    usually visible anyway; restricting the masking task to context forces
    the reconstruction signal to flow through genotype positions.
    """
    masked = input_ids.clone()
    eligible = input_ids < TOKEN_HOM_REF
    mask = eligible & (
        torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
        < mask_fraction
    )
    # -100 at KEPT positions, original ids at MASKED positions (ignore_index
    # semantics): CE scores exactly the masked set. NB: the F-14 run executed
    # with this fill inverted (-100 at masked), i.e. it scored the VISIBLE
    # tokens — a degenerate identity-copy objective. Discovered while building
    # F-14a (2026-09-28); F-14 evidence annotated, not re-run (PM-006 addendum).
    targets = input_ids.masked_fill(~mask, -100)
    masked[mask] = mask_token_id
    return masked, targets


class MambaMLM(nn.Module):
    """Wraps a MambaSequenceClassifier's encoder with a token-prediction head.

    The head is a tied-weight linear projection of d_model -> vocab that is
    discarded after pretraining; the encoder transfers exactly.
    """

    def __init__(self, classifier: nn.Module, vocab_size: int = PRETRAIN_VOCAB) -> None:
        super().__init__()
        # MASK token id (VOCAB_SIZE) needs one embedding row past the
        # classifier's supervised vocab — grow the table before training.
        if hasattr(classifier, "ensure_vocab"):
            classifier.ensure_vocab(vocab_size)
        self.classifier = classifier
        self.lm_head = nn.Linear(classifier.d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, L)
        h = self.classifier.embed(input_ids)                    # (B, L, d)
        return self.lm_head(h)                                  # (B, L, vocab)

    def pretrain_loss(self, input_ids: torch.Tensor, generator: torch.Generator | None = None,
                      mask_context_only: bool = False) -> torch.Tensor:
        """CE over masked positions only (scores just those rows' logits).

        Masking ~15% of 504 tokens leaves ~76 scored positions per batch
        element, so logits are gathered to (n_masked, vocab) before the
        softmax — the full (B, L, vocab) tensor is never materialized.

        mask_context_only=True routes to mask_context_tokens (F-14a: the
        genotype tokens 4096..4098 are never masked, so reconstruction of
        masked context must use them — targets the recorded F-14 root cause
        of context memorization).
        """
        masker = mask_context_tokens if mask_context_only else mask_tokens
        masked, targets = masker(input_ids, generator=generator)
        h = self.classifier.embed(masked)                       # (B, L, d)
        sel = targets != -100
        if not bool(sel.any()):
            # No masked positions (possible under mask_context_only when a
            # batch holds only genotype tokens): return a graph-connected
            # zero so backward stays well-defined (plain CE would be NaN).
            return h.sum() * 0.0
        h_sel = h[sel]                                          # (M, d)
        logits = self.lm_head(h_sel)                            # (M, vocab)
        return F.cross_entropy(logits, targets[sel], ignore_index=-100)
