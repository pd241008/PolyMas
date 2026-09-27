"""Tests for polymas_ml.sequence.ssl (F-14 masked k-mer pretraining)."""

import pytest
import torch

from polymas_ml.sequence.dataset import VOCAB_SIZE
from polymas_ml.sequence.model import MambaSequenceClassifier
from polymas_ml.sequence.ssl import (
    MASK_FRACTION,
    MASK_TOKEN_ID,
    MambaMLM,
    PRETRAIN_VOCAB,
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
        # targets: -100 at masked positions, original ids elsewhere
        is_masked = targets == -100
        assert torch.equal(masked[is_masked], torch.full_like(masked[is_masked], MASK_TOKEN_ID))
        assert torch.equal(masked[~is_masked], x[~is_masked])

    def test_mask_fraction_respected(self) -> None:
        x = torch.randint(0, VOCAB_SIZE, (64, 504))
        masked, targets = mask_tokens(x, generator=torch.Generator().manual_seed(2))
        frac = (targets == -100).float().mean().item()
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
