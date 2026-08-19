.PHONY: install data data-offline smoke train sample test serve docker docker-up docker-down clean

install:
	pip install -r requirements.txt

data:
	python -m src.data --dataset tiny-shakespeare

data-offline:
	python -m src.data --dataset tiny-shakespeare --offline

# tiny-CPU offline smoke: no downloads, no GPU, no keys.
# trains a tiny char GPT a few hundred steps on the bundled corpus, samples,
# and serves it through the FastAPI /generate endpoint in-process.
smoke:
	python scripts/smoke.py

train:
	python -m src.train --config configs/tiny-shakespeare.yaml

sample:
	python -m src.sample --ckpt out/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200

test:
	pytest tests/ -v

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -f deploy/Dockerfile -t mini-gpt:dev .

docker-up:
	docker compose -f deploy/docker-compose.yml up --build

docker-down:
	docker compose -f deploy/docker-compose.yml down

clean:
	rm -rf __pycache__ .pytest_cache out/ mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} +
