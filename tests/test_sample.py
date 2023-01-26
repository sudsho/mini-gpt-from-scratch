"""tests for the generate() method."""

import torch
import pytest

from src.model import GPT, GPTConfig


@pytest.fixture
def tiny_model():
    cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    return GPT(cfg), cfg


def test_generate_extends_sequence(tiny_model):
    model, cfg = tiny_model
    model.eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
    # all generated ids must be in vocab range
    assert int(out.max()) < cfg.vocab_size
    assert int(out.min()) >= 0


def test_generate_does_not_exceed_block_size(tiny_model):
    """when prompt is longer than block_size, model should crop and still work."""
    model, cfg = tiny_model
    long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 4))
    out = model.generate(long_prompt, max_new_tokens=3)
    assert out.shape[1] == cfg.block_size + 4 + 3


def test_generate_top_k(tiny_model):
    """with top_k=1 (greedy), output should be deterministic across runs."""
    model, _ = tiny_model
    model.eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    o1 = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=1)
    # different seed should not matter for greedy
    torch.manual_seed(99)
    o2 = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(o1, o2)
