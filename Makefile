# fraudnet build targets. Every target runs through uv so the same interpreter is used everywhere.
UV ?= uv
PY := $(UV) run python
SEED ?= 42
EXTRA_SEEDS ?= 7,2026
HOST ?= 127.0.0.1
PORT ?= 8772
export OMP_NUM_THREADS ?= 2

.PHONY: help setup lint format test bench bench-quick check tune train serve leakage docker-build docker-run clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the environment and simulate the dataset (cached under data/)
	$(UV) sync
	$(PY) -m fraudnet.cli simulate --seed $(SEED)

lint: ## Ruff lint and format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Apply ruff formatting and safe fixes
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

test: ## Run the test suite
	$(UV) run pytest -q

bench: ## Full benchmark: results.json and results.md, README tables, threshold gate
	$(PY) -m fraudnet.bench --seed $(SEED) --seeds $(EXTRA_SEEDS) --update-readme --check

bench-quick: ## The CI benchmark on the smaller world, with its threshold gate
	$(PY) -m fraudnet.bench --quick --check

check: ## Re-check benchmarks/results.json against benchmarks/thresholds.json
	$(PY) -m fraudnet.bench --check-only

tune: ## Validation-only hyperparameter sweep (never reads test labels)
	$(PY) -m fraudnet.tune --seed $(SEED)

train: ## Fit the serving bundle into artifacts/
	$(PY) -m fraudnet.cli train --seed $(SEED)

serve: ## Serve the API and dashboard on http://$(HOST):$(PORT) (trains first if artifacts/ is empty)
	$(PY) -m fraudnet.cli serve --seed $(SEED) --host $(HOST) --port $(PORT)

leakage: ## Run the point-in-time checker on the first 30 days
	$(PY) -m fraudnet.cli leakage --seed $(SEED)

docker-build: ## Build the container image (trains the bundle inside the build)
	docker build -t fraudnet:local .

docker-run: ## Run the container on port 8772
	docker run --rm -p 8772:8772 fraudnet:local

clean: ## Remove caches, artifacts and benchmark outputs (keeps the simulated data)
	rm -rf .pytest_cache .ruff_cache .hypothesis artifacts/*.pkl benchmarks/ci-results.*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
