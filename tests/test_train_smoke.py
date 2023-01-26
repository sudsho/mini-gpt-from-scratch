"""tiny smoke test for one training step.

this just checks that one optim step on a randomly initialised model reduces
the loss on a fixed batch. with the conftest seed it's deterministic.
"""

import torch

from src.model import GPT, GPTConfig


def test_one_step_decreases_loss():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _, loss0 = model(x, y)
    loss0_val = loss0.item()

    for _ in range(5):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    _, loss1 = model(x, y)
    assert loss1.item() < loss0_val, f"loss did not decrease: {loss0_val} -> {loss1.item()}"
