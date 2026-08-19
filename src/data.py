"""data utilities: download, char vocab, train/val split."""

import os
import shutil
import argparse
import pickle
import numpy as np


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# a small public-domain char corpus bundled in the repo so the demo and the
# smoke test run with no network. lives at <repo>/data/tiny_corpus.txt.
BUNDLED_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "tiny_corpus.txt",
)


def download_tiny_shakespeare(data_dir, offline=None):
    """fetch the tiny shakespeare corpus.

    if offline is True (or the MINI_GPT_OFFLINE env var is set) we skip the
    network entirely and copy the bundled public-domain corpus instead. if a
    download is attempted and fails, we fall back to the bundled corpus so the
    pipeline still runs with no keys/downloads.
    """
    if offline is None:
        offline = os.environ.get("MINI_GPT_OFFLINE", "") not in ("", "0", "false")

    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "input.txt")
    if os.path.exists(raw_path):
        print(f"already have {raw_path}")
        return raw_path

    if offline:
        print(f"offline mode: using bundled corpus {BUNDLED_CORPUS}")
        shutil.copyfile(BUNDLED_CORPUS, raw_path)
        return raw_path

    try:
        import requests
        print(f"downloading tiny shakespeare to {raw_path}")
        r = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
        r.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    except Exception as e:  # noqa: BLE001  (network down, no keys, etc.)
        print(f"download failed ({e}); falling back to bundled corpus {BUNDLED_CORPUS}")
        shutil.copyfile(BUNDLED_CORPUS, raw_path)
    return raw_path


class CharTokenizer:
    """very small char-level tokenizer."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def from_text(cls, text):
        vocab = sorted(list(set(text)))
        return cls(vocab)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["vocab"])


def prepare_tiny_shakespeare(data_dir, val_frac=0.1, offline=None):
    raw_path = download_tiny_shakespeare(data_dir, offline=offline)
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    print(f"vocab size = {tokenizer.vocab_size}")

    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    n = len(ids)
    split = int(n * (1.0 - val_frac))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    tok_path = os.path.join(data_dir, "tokenizer.pkl")
    tokenizer.save(tok_path)

    print(f"train tokens = {len(train_ids):,}, val tokens = {len(val_ids):,}")
    return train_path, val_path, tok_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiny-shakespeare")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--offline", action="store_true",
                    help="skip the network and use the bundled corpus")
    args = ap.parse_args()

    if args.dataset == "tiny-shakespeare":
        data_dir = args.data_dir or "data/tiny-shakespeare"
        prepare_tiny_shakespeare(
            data_dir, val_frac=args.val_frac,
            offline=True if args.offline else None,
        )
    else:
        raise ValueError(f"unknown dataset {args.dataset}")


if __name__ == "__main__":
    main()
