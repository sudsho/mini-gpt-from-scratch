"""FastAPI inference service for the mini-gpt model."""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import GPT, GPTConfig
from src.data import CharTokenizer
from src.sample import load_tokenizer


CKPT_PATH = os.environ.get("MINI_GPT_CKPT", "out/tiny-shakespeare/ckpt.pt")
TOK_KIND = os.environ.get("MINI_GPT_TOKENIZER", "char")
TOK_PATH = os.environ.get(
    "MINI_GPT_TOKENIZER_PATH",
    "data/tiny-shakespeare/tokenizer.pkl",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI(title="mini-gpt", version="0.1.0")


_state = {"model": None, "config": None, "tokenizer": None}


@app.on_event("startup")
def _load():
    if not os.path.exists(CKPT_PATH):
        # serve will raise on /generate; let /health still respond.
        return
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _state["model"] = model
    _state["config"] = cfg
    _state["tokenizer"] = load_tokenizer(TOK_KIND, TOK_PATH)


class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(100, ge=1, le=1024)
    temperature: float = Field(0.8, gt=0.0, le=5.0)
    top_k: int = Field(40, ge=0, le=1000)
    seed: int = Field(1337, ge=0)


class GenResponse(BaseModel):
    prompt: str
    completion: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail=f"checkpoint not found at {CKPT_PATH}")

    torch.manual_seed(req.seed)
    tok = _state["tokenizer"]
    model = _state["model"]

    prompt_ids = tok.encode(req.prompt)
    if len(prompt_ids) == 0:
        raise HTTPException(status_code=400, detail="empty prompt after tokenization")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    # crop prompt to the model's block size up front so we don't crash inside generate
    block_size = _state["config"].block_size
    if idx.size(1) > block_size:
        idx = idx[:, -block_size:]

    top_k = req.top_k if req.top_k > 0 else None
    with torch.no_grad():
        out = model.generate(idx, req.max_new_tokens, req.temperature, top_k)

    full = tok.decode(out[0].tolist())
    # only return the newly generated suffix, not the (possibly cropped) prompt
    completion = full[len(tok.decode(prompt_ids)):]
    return GenResponse(prompt=req.prompt, completion=completion)
