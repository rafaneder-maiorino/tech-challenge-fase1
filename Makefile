.PHONY: install lint test run clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	python -m pytest tests/ -v

run:
	uvicorn src.api.main:app --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete