"""Tests for polymas_ml.sequence.ssl (F-14 masked k-mer pretraining)."""

import pytest
import torch

from polymas_ml.sequence.dataset import (
    TOKEN_HOM_ALT,
    TOKEN_HET,
    TOKEN_HOM_REF,
    VOCAB_SIZE,
)
from polymas_ml.sequence.model import MambaSequenceClassifier
from polymas_ml.sequence.ssl import (
    MASK_FRACTION,
    MASK_TOKEN_ID,
    MambaMLM,
    PRETRAIN_VOCAB,
    mask_context_tokens,
    mask_tokens,
)


def _tiny_classifier() -> MambaSequenceClassifier:
    torch.manual_seed(0)
    return MambaSequenceClassifier(
        vocab_size=VOCAB_SIZE,
        n_diseases=7,
        d_model=32,
        n_layers=2,
    )


class TestMaskTokens:
    def test_shapes_and_vocab_bounds(self) -> None:
        x = torch.randint(0, VOCAB_SIZE, (4, 504))
        masked, targets = mask_tokens(x, generator=torch.Generator().manual_seed(1))
        assert masked.shape == x.shape
        assert targets.shape == x.shape
        assert (masked < PRETRAIN_VOCAB).all()
        # Convention (documented): original ids at MASKED positions, -100 at
        # kept positions, so CE with ignore_index=-100 scores the masked set.
        is_masked = targets != -100
        assert torch.equal(masked[is_masked],
                           torch.full_like(masked[is_masked], MASK_TOKEN_ID))
        assert torch.equal(targets[is_masked], x[is_masked])
        assert torch.equal(masked[~is_masked], x[~is_masked])

    def test_mask_fraction_respected(self) -> None:
        x = torch.randint(0, VOCAB_SIZE, (64, 504))
        masked, targets = mask_tokens(x, generator=torch.Generator().manual_seed(2))
        frac = (targets != -100).float().mean().item()
        assert 0.10 < frac < 0.20  # 15% +- sampling noise at 32k tokens

    def test_generator_determinism(self) -> None:
        x = torch.randint(0, VOCAB_SIZE, (8, 504))
        m1, t1 = mask_tokens(x, generator=torch.Generator().manual_seed(7))
        m2, t2 = mask_tokens(x, generator=torch.Generator().manual_seed(7))
        assert torch.equal(m1, m2)
        assert torch.equal(t1, t2)

    def test_never_masks_everything(self) -> None:
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        for seed in range(20):
            _, targets = mask_tokens(x, generator=torch.Generator().manual_seed(seed))
            assert (targets == -100).any()  # at least one masked
            assert (targets != -100).any()  # never all masked


class TestMaskContextTokens:
    """F-14a variant: context k-mers masked, genotype tokens always visible."""

    def test_genotype_tokens_never_masked(self) -> None:
        # Sequence of ONLY genotype tokens (worst case) -> nothing scored.
        x = torch.full((8, 504), TOKEN_HOM_ALT)
        for seed in range(20):
            _, targets = mask_context_tokens(
                x, generator=torch.Generator().manual_seed(seed))
            assert not (targets != -100).any()

    def test_only_context_positions_masked(self) -> None:
        # Mixed sequence: scored (masked) positions must be context ids only.
        x = torch.randint(0, VOCAB_SIZE, (16, 504))
        x[:, ::63] = TOKEN_HET  # sprinkle genotype tokens
        masked, targets = mask_context_tokens(
            x, generator=torch.Generator().manual_seed(5))
        is_masked = targets != -100
        assert is_masked.any()  # context was available -> some scored
        assert torch.equal(masked[is_masked],
                           torch.full_like(masked[is_masked], MASK_TOKEN_ID))
        assert (x[is_masked] < TOKEN_HOM_REF).all()  # eligibility held
        # genotype positions: input untouched and never scored (target -100)
        gen = x >= TOKEN_HOM_REF
        assert torch.equal(masked[gen], x[gen])
        assert (targets[gen] == -100).all()

    def test_mask_fraction_scope(self) -> None:
        # 15% of CONTEXT tokens, not of all tokens.
        x = torch.randint(0, VOCAB_SIZE, (64, 504))
        x[:, ::7] = TOKEN_HOM_REF  # ~14% genotype tokens
        _, targets = mask_context_tokens(
            x, generator=torch.Generator().manual_seed(6))
        n_masked = int((targets != -100).sum())
        n_context = int((x < TOKEN_HOM_REF).sum())
        frac = n_masked / n_context
        assert 0.10 < frac < 0.20
        assert n_masked < n_context

    def test_pretrain_loss_routes_to_context_masker(self) -> None:
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.full((2, 504), TOKEN_HOM_ALT)  # no context -> no masking
        loss_ctx = model.pretrain_loss(
            x, generator=torch.Generator().manual_seed(8), mask_context_only=True)
        loss_std = model.pretrain_loss(
            x, generator=torch.Generator().manual_seed(8))
        assert float(loss_ctx) == 0.0   # nothing scored under context-only
        assert float(loss_std) > 0.0    # standard path still masks everything
        assert torch.isfinite(loss_ctx) and torch.isfinite(loss_std)

    def test_zero_mask_batch_backward_is_finite(self) -> None:
        # Empty-CE NaN guard: context-only masking of an all-genotype batch
        # returns a graph-connected zero; backward populates grads.
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.full((2, 504), TOKEN_HET)
        loss = model.pretrain_loss(
            x, generator=torch.Generator().manual_seed(10), mask_context_only=True)
        assert float(loss) == 0.0 and torch.isfinite(loss)
        loss.backward()
        assert clf.embedding.weight.grad is not None
        assert torch.isfinite(clf.embedding.weight.grad).all()

    def test_default_behavior_unchanged(self) -> None:
        # The F-14 protocol path (no flag) must be bit-identical to before.
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        gen = torch.Generator().manual_seed(9)
        loss_flag_off = model.pretrain_loss(x, generator=gen)
        masked, targets = mask_tokens(
            x, generator=torch.Generator().manual_seed(9))
        sel = targets != -100
        import torch.nn.functional as F
        logits = model(masked)
        h_sel = model.classifier.embed(masked)[sel]
        loss_manual = F.cross_entropy(model.lm_head(h_sel), targets[sel],
                                      ignore_index=-100)
        assert torch.allclose(loss_flag_off, loss_manual, rtol=1e-4, atol=1e-5)


class TestMambaMLM:
    def test_forward_shape(self) -> None:
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        logits = model(x)
        assert logits.shape == (2, 504, PRETRAIN_VOCAB)

    def test_pretrain_loss_finite_and_mask_scoped(self) -> None:
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        loss = model.pretrain_loss(x, generator=torch.Generator().manual_seed(3))
        assert torch.isfinite(loss)

    def test_pretrain_loss_matches_manual_masked_ce(self) -> None:
        # Masked-only scoring equals CE over the full grid with ignore_index.
        import torch.nn.functional as F
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        gen = torch.Generator().manual_seed(11)
        loss_sel = model.pretrain_loss(x, generator=gen)
        masked, targets = mask_tokens(x, generator=torch.Generator().manual_seed(11))
        logits = model(masked)
        loss_full = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        assert torch.allclose(loss_sel, loss_full, rtol=1e-4, atol=1e-5)

    def test_mask_token_embedding_created_on_demand(self) -> None:
        clf = _tiny_classifier()
        assert clf.embedding.num_embeddings == VOCAB_SIZE
        clf.ensure_vocab(PRETRAIN_VOCAB)
        assert clf.embedding.num_embeddings == PRETRAIN_VOCAB
        assert torch.isfinite(clf.embedding.weight).all()

    def test_encoder_transfers_exactly(self) -> None:
        clf = _tiny_classifier()
        w_orig = clf.embedding.weight.detach().clone()
        model = MambaMLM(clf)
        assert model.classifier is clf
        # growing the table preserved every original row exactly
        assert clf.embedding.num_embeddings == PRETRAIN_VOCAB
        assert torch.equal(clf.embedding.weight.detach()[:VOCAB_SIZE], w_orig)
        # pretrain step produces grads on the encoder path
        x = torch.randint(0, VOCAB_SIZE, (2, 504))
        model.pretrain_loss(x, generator=torch.Generator().manual_seed(4)).backward()
        assert clf.embedding.weight.grad is not None
        # forward/backward never mutates weight VALUES (only grads)
        assert torch.equal(clf.embedding.weight.detach()[:VOCAB_SIZE], w_orig)

    def test_lm_head_untied_from_classifier(self) -> None:
        clf = _tiny_classifier()
        model = MambaMLM(clf)
        assert model.lm_head.weight.shape == (PRETRAIN_VOCAB, clf.d_model)
        # head params are disjoint from the classifier's output head
        head_params = {id(p) for p in model.lm_head.parameters()}
        clf_params = {id(p) for p in clf.parameters()}
        assert not (head_params & clf_params)
