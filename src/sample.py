"""sample text from a trained checkpoint."""

import os
import argparse
import pickle
import torch

from src.model import GPT, GPTConfig
from src.data import CharTokenizer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    config = GPTConfig(**cfg_dict)
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.pkl, default: data/tiny-shakespeare/tokenizer.pkl")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_path = args.tokenizer or "data/tiny-shakespeare/tokenizer.pkl"
    tokenizer = CharTokenizer.load(tok_path)

    model, _ = load_model(args.ckpt, device)

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
