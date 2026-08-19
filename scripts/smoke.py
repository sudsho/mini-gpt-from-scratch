"""tiny-CPU offline smoke for mini-gpt-from-scratch.

no downloads, no GPU, no API keys. everything runs on CPU against the small
public-domain char corpus bundled at data/tiny_corpus.txt.

what it does, in order:
  1. shape asserts on the attention block and a full transformer block
  2. build a char tokenizer from the bundled corpus
  3. train a tiny char-level GPT for a few hundred steps and check the loss drops
  4. sample a short continuation (should look vaguely text-like)
  5. save a checkpoint + tokenizer, then hit the FastAPI /generate endpoint
     in-process (TestClient) to prove serving works end-to-end

run:  python scripts/smoke.py     (or:  make smoke)
"""

import os
import sys
import pickle

import numpy as np
import torch

# make the repo importable when run as a plain script (python scripts/smoke.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import GPT, GPTConfig, CausalSelfAttention, Block
from src.data import CharTokenizer, BUNDLED_CORPUS

# force everything onto CPU regardless of what torch thinks is available
DEVICE = "cpu"
torch.manual_seed(1337)
np.random.seed(1337)

OUT_DIR = os.path.join(REPO_ROOT, "out", "smoke")


def check_shapes():
    """block / attention shape asserts (part of the smoke goal)."""
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)
    x = torch.randn(2, 16, cfg.n_embd)

    attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size)
    assert attn(x).shape == x.shape, "attention changed the tensor shape"

    block = Block(cfg.n_embd, cfg.n_head, cfg.block_size)
    assert block(x).shape == x.shape, "block changed the tensor shape"

    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 10))
    logits, _ = model(idx)
    assert logits.shape == (4, 10, cfg.vocab_size), "bad logits shape"

    # causal mask must not leak the future into the past
    model.eval()
    a = torch.randint(0, cfg.vocab_size, (1, 16))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 7) % cfg.vocab_size
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :8, :], lb[:, :8, :], atol=1e-6), "causal mask leaks"
    print("[1/5] shape + causal-mask asserts passed")


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train_tiny(steps=300):
    with open(BUNDLED_CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer.from_text(text)
    print(f"[2/5] built char tokenizer: vocab_size={tok.vocab_size}, corpus_chars={len(text)}")

    ids = np.array(tok.encode(text), dtype=np.uint16)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GPT(cfg).to(DEVICE)
    model.train()
    print(f"      tiny GPT: {model.num_params():,} params, "
          f"n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} block_size={cfg.block_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99))

    batch_size = 32
    first_loss = None
    last_loss = None
    for step in range(steps):
        x, y = get_batch(ids, cfg.block_size, batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 50 == 0 or step == steps - 1:
            print(f"      step {step:4d}  loss {loss.item():.4f}")

    print(f"[3/5] trained {steps} steps: loss {first_loss:.4f} -> {last_loss:.4f}")
    assert last_loss < first_loss, f"loss did not drop: {first_loss:.4f} -> {last_loss:.4f}"
    return model, cfg, tok


def sample(model, tok, prompt="ROMEO:", max_new_tokens=200):
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=20)
    text = tok.decode(out[0].tolist())
    print("[4/5] sample from the trained tiny model:")
    print("-" * 60)
    print(text)
    print("-" * 60)
    return text


def save_ckpt(model, cfg, tok):
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    tok_path = os.path.join(OUT_DIR, "tokenizer.pkl")
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckpt_path)
    tok.save(tok_path)
    return ckpt_path, tok_path


def check_api(ckpt_path, tok_path):
    # point the app at the smoke checkpoint before importing it
    os.environ["MINI_GPT_CKPT"] = ckpt_path
    os.environ["MINI_GPT_TOKENIZER"] = "char"
    os.environ["MINI_GPT_TOKENIZER_PATH"] = tok_path

    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200, h.text
        assert h.json()["model_loaded"] is True, "api did not load the smoke checkpoint"

        r = client.post("/generate", json={
            "prompt": "ROMEO:",
            "max_new_tokens": 60,
            "temperature": 0.8,
            "top_k": 20,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body["completion"], str) and len(body["completion"]) > 0
    print("[5/5] FastAPI /health + /generate served the tiny model:")
    print(f"      prompt='{body['prompt']}' completion={body['completion']!r}")


def main():
    print("=== mini-gpt tiny-CPU offline smoke ===")
    check_shapes()
    model, cfg, tok = train_tiny(steps=300)
    sample(model, tok)
    ckpt_path, tok_path = save_ckpt(model, cfg, tok)
    check_api(ckpt_path, tok_path)
    print("=== SMOKE OK ===")


if __name__ == "__main__":
    main()
