"""data utilities. for now: download Tiny Shakespeare and build a char vocab."""

import os
import argparse
import requests
import numpy as np


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(raw_path):
        print(f"downloading tiny shakespeare to {raw_path}")
        r = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
        r.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    else:
        print(f"already have {raw_path}")
    return raw_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiny-shakespeare")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if args.dataset == "tiny-shakespeare":
        data_dir = args.data_dir or "data/tiny-shakespeare"
        download_tiny_shakespeare(data_dir)
    else:
        raise ValueError(f"unknown dataset {args.dataset}")


if __name__ == "__main__":
    main()
