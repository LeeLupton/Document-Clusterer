.RECIPEPREFIX := >
.PHONY: format lint test all

format:
>ruff format document_clusterer tests

lint:
>ruff check .
>mypy document_clusterer tests

test:
>PYTHONPATH=. pytest

all: format lint test
