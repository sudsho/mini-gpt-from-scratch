.PHONY: install data train sample test serve docker clean

install:
	pip install -r requirements.txt

data:
	python -m src.data --dataset tiny-shakespeare

train:
	python -m src.train --config configs/tiny-shakespeare.yaml

sample:
	python -m src.sample --ckpt out/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200

test:
	pytest tests/ -v

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -t mini-gpt .

clean:
	rm -rf __pycache__ .pytest_cache out/ mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} +
