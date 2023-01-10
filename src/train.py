"""training loop for the GPT.

reads YAML config, trains, evaluates on val set, saves checkpoints.
"""

import os
import math
import time
import argparse
import pickle
import yaml
import numpy as np
import torch

from src.model import GPT, GPTConfig


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 1337))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # data
    data_dir = cfg["data_dir"]
    train_data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)

    with open(os.path.join(data_dir, "tokenizer.pkl"), "rb") as f:
        meta = pickle.load(f)
    vocab_size = len(meta["vocab"])
    print(f"vocab size = {vocab_size}, train tokens = {len(train_data):,}")

    # model
    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
        bias=cfg.get("bias", False),
    )
    model = GPT(gpt_cfg).to(device)
    print(f"model has {model.num_params():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for it in range(cfg["max_iters"]):
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg.get("log_interval", 10) == 0:
            print(f"iter {it} loss {loss.item():.4f}")

    torch.save(
        {"model": model.state_dict(), "config": vars(gpt_cfg)},
        os.path.join(out_dir, "ckpt.pt"),
    )
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
