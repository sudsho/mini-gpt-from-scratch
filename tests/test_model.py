"""shape tests for the GPT model."""

import torch
import pytest

from src.model import GPT, GPTConfig, CausalSelfAttention, Block


@pytest.fixture
def small_cfg():
    return GPTConfig(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_attention_output_shape(small_cfg):
    cfg = small_cfg
    attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size)
    x = torch.randn(2, 16, cfg.n_embd)
    y = attn(x)
    assert y.shape == x.shape


def test_block_shape(small_cfg):
    cfg = small_cfg
    block = Block(cfg.n_embd, cfg.n_head, cfg.block_size)
    x = torch.randn(2, 16, cfg.n_embd)
    y = block(x)
    assert y.shape == x.shape


def test_gpt_forward_shape(small_cfg):
    model = GPT(small_cfg)
    idx = torch.randint(0, small_cfg.vocab_size, (4, 10))
    logits, loss = model(idx)
    assert logits.shape == (4, 10, small_cfg.vocab_size)
    assert loss is None


def test_gpt_forward_with_targets(small_cfg):
    model = GPT(small_cfg)
    idx = torch.randint(0, small_cfg.vocab_size, (4, 10))
    targets = torch.randint(0, small_cfg.vocab_size, (4, 10))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 10, small_cfg.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_causal_mask_no_future_leak(small_cfg):
    """changing future tokens should not affect past predictions (left side)."""
    model = GPT(small_cfg)
    model.eval()
    torch.manual_seed(0)
    idx_a = torch.randint(0, small_cfg.vocab_size, (1, 16))
    idx_b = idx_a.clone()
    idx_b[0, 8:] = (idx_b[0, 8:] + 7) % small_cfg.vocab_size  # perturb the future
    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)
    # left half (indices 0..7) must be identical
    assert torch.allclose(logits_a[:, :8, :], logits_b[:, :8, :], atol=1e-6)


def test_block_size_assertion(small_cfg):
    model = GPT(small_cfg)
    too_long = torch.zeros(1, small_cfg.block_size + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        model(too_long)
