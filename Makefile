.PHONY: install dev test lint format run docker-build docker-run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

docker-build:
	docker build -t limit-order-book .

docker-run:
	docker run -p 8501:8501 limit-order-book

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache .ruff_cache dist build
