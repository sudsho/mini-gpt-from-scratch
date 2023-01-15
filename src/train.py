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
import mlflow

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


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def cosine_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


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

    # mlflow
    run_name = cfg.get("run_name", os.path.basename(args.config))
    mlflow.set_experiment(cfg.get("experiment_name", "mini-gpt"))
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))})

    t0 = time.time()
    best_val = float("inf")
    for it in range(cfg["max_iters"]):
        # set LR
        lr = cosine_lr(
            it,
            cfg.get("warmup_iters", 100),
            cfg.get("lr_decay_iters", cfg["max_iters"]),
            cfg["learning_rate"],
            cfg.get("min_lr", cfg["learning_rate"] / 10),
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if it % cfg.get("log_interval", 10) == 0:
            print(f"iter {it} loss {loss.item():.4f} lr {lr:.5f}")
            mlflow.log_metric("train_loss_step", loss.item(), step=it)
            mlflow.log_metric("lr", lr, step=it)

        if it > 0 and it % cfg.get("eval_interval", 250) == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                cfg["block_size"], cfg["batch_size"],
                cfg.get("eval_iters", 100), device,
            )
            print(f"eval iter {it} train {losses['train']:.4f} val {losses['val']:.4f}")
            mlflow.log_metric("train_loss", losses["train"], step=it)
            mlflow.log_metric("val_loss", losses["val"], step=it)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": vars(gpt_cfg)},
                    os.path.join(out_dir, "ckpt.pt"),
                )

    print(f"done in {time.time() - t0:.1f}s, best val {best_val:.4f}")
    mlflow.log_metric("best_val_loss", best_val)
    mlflow.end_run()


if __name__ == "__main__":
    main()
