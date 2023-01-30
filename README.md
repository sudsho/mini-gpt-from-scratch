# mini-gpt-from-scratch

build a small decoder-only GPT from scratch in PyTorch. train on Tiny Shakespeare (char-level) first, then on a small slice of OpenWebText with a BPE tokenizer.

inspired by Andrej Karpathy's nanoGPT (released early Jan 2023). this is my own re-implementation of the same ideas, kept small enough to train on a single GPU.

## why

i wanted to actually understand the transformer block by writing it line-by-line. the goal is not to beat any benchmark, just to:

1. write the model in plain PyTorch (no `nn.MultiheadAttention`)
2. train on Tiny Shakespeare and then on a small OpenWebText subset
3. sample from the model with temperature / top-k
4. wrap inference in a small FastAPI service so it can be deployed

## architecture

standard decoder-only transformer:

- token + position embeddings
- N x Block (causal multi-head self-attention + 2-layer MLP, both pre-LayerNorm)
- final LayerNorm + tied lm_head
- AdamW + cosine LR schedule, gradient clipping

defaults (config/tiny-shakespeare.yaml): 6 layers, 6 heads, 384 dim, block size 256, char-level vocab.

## datasets

- Tiny Shakespeare (~1MB) — `src/data.py` downloads from karpathy's repo, char-level vocab (~65 tokens).
- OpenWebText subset — small slice tokenized with `tiktoken` GPT-2 BPE.

## quickstart

```bash
pip install -r requirements.txt
python -m src.data --dataset tiny-shakespeare
python -m src.train --config configs/tiny-shakespeare.yaml
python -m src.sample --ckpt out/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200
```

## inference api

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "ROMEO:", "max_new_tokens": 100, "temperature": 0.8, "top_k": 40}'
```

## training curves

trained on a single RTX 3090 for ~15 min on Tiny Shakespeare. final val loss ~1.46. see `notebooks/explore.ipynb` for loss curves and attention pattern visualizations from the MLflow run.

### sample after 5000 iters

```
ROMEO:
What's't, Of mine to be the morrow that I will not so much:
Forsworn the loyal, blow off the seal of my heart.
What is the wars then? was thy soul to a soul,
Where dost thou lay this hour-eyes go thou with my queen?
```

(this is a small char-level model, so output is shakespearean noise, not coherent).

## docker

```bash
# build the image and run the api
make docker
docker run -p 8000:8000 \
    -v $(pwd)/out:/app/out:ro \
    -v $(pwd)/data:/app/data:ro \
    mini-gpt:dev

# or with docker compose
make docker-up
```

The image uses the CPU-only torch wheel so it stays around 1GB. For training a real model you want a CUDA base image, see `deploy/Dockerfile` for the pattern.

## tests

```bash
pytest tests/ -v
```

`tests/conftest.py` seeds python/numpy/torch before every test. Coverage:

- `test_data.py` - char tokenizer roundtrip and save/load
- `test_model.py` - shape tests, causal mask leak check, block size assertion
- `test_sample.py` - generate() shape, cropping when prompt > block size, greedy determinism
- `test_api.py` - FastAPI routes via TestClient (health, validation, 503 when no ckpt)
- `test_train_smoke.py` - one-batch optim step actually decreases loss

## ci

A GitHub Actions workflow that runs the test suite on push/PR is in `ci/test.yml.example`. Move it to `.github/workflows/test.yml` after granting the `workflow` scope to the deploy token.

## license

MIT (see LICENSE).
