"""prepare a small slice of OpenWebText with GPT-2 BPE.

uses huggingface datasets to stream a subset, tokenizes with tiktoken (gpt2),
writes train.bin / val.bin (uint16) and meta.pkl with vocab_size.

usage:
    python -m src.prepare_owt --num-docs 5000 --out-dir data/openwebtext
"""

import os
import argparse
import pickle
import numpy as np

# tiktoken is heavy to load lazily so import inside main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=5000,
                    help="number of openwebtext documents to use (small slice)")
    ap.add_argument("--out-dir", default="data/openwebtext")
    ap.add_argument("--val-frac", type=float, default=0.005)
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset  # noqa: WPS433  (huggingface datasets)

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"streaming OpenWebText, taking first {args.num_docs} docs")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    all_ids = []
    eot = enc.eot_token  # GPT-2 uses 50256 as <|endoftext|>
    for i, ex in enumerate(ds):
        if i >= args.num_docs:
            break
        ids = enc.encode_ordinary(ex["text"])
        ids.append(eot)
        all_ids.extend(ids)
        if i % 500 == 0:
            print(f"  doc {i} -> total tokens {len(all_ids):,}")

    arr = np.array(all_ids, dtype=np.uint16)  # gpt2 vocab fits in uint16 (50257)
    n = len(arr)
    split = int(n * (1.0 - args.val_frac))
    arr[:split].tofile(os.path.join(args.out_dir, "train.bin"))
    arr[split:].tofile(os.path.join(args.out_dir, "val.bin"))

    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "tokenizer": "tiktoken-gpt2"}, f)

    print(f"wrote {n:,} tokens, vocab size {enc.n_vocab}")


if __name__ == "__main__":
    main()
