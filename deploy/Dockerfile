# CPU-only image for serving the mini-gpt model.
# for training a real model you'd want a CUDA base image instead.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install just the runtime deps to keep the image smaller
COPY deploy/requirements-serve.txt /app/requirements-serve.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-serve.txt

COPY src/ /app/src/
COPY configs/ /app/configs/

# the checkpoint is mounted in at runtime via a bind mount or volume
ENV MINI_GPT_CKPT=/app/out/ckpt.pt \
    MINI_GPT_TOKENIZER=char \
    MINI_GPT_TOKENIZER_PATH=/app/data/tiny-shakespeare/tokenizer.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
