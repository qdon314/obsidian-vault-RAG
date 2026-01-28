.PHONY: help index index-dummy ask ask-dummy results tail-logs clean-index \
        test lint fmt typecheck env-check

# ---- Python interpreter (pin to micromamba env) ----
MAMBA_ROOT_PREFIX ?= /Users/quentindonnelly/micromamba
CONDA_ENV ?= rag-obsidian
PYTHON := $(MAMBA_ROOT_PREFIX)/envs/$(CONDA_ENV)/bin/python

ifeq ($(wildcard $(PYTHON)),)
$(error Could not find PYTHON at $(PYTHON). Check MAMBA_ROOT_PREFIX / CONDA_ENV.)
endif
# -----------------------------------------------------

ARTIFACTS_DIR ?= artifacts
INDEX ?= obsidian
CORPUS ?= /Users/quentindonnelly/Documents/Personal\ \&\ Professional
QUERY ?= What is this project about?
NUM_LOGS ?= 20

help:  ## Show available commands
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------

index:  ## Build index from corpus (JSONL store)
	$(PYTHON) scripts/build_index.py \
		--index-name $(INDEX) \
		--corpus "$(CORPUS)" \
		--artifacts-dir $(ARTIFACTS_DIR)

index-dummy:  ## Build index using DummyEmbedder
	$(PYTHON) scripts/build_index.py \
		--index-name $(INDEX) \
		--corpus "$(CORPUS)" \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--use-dummy-embeddings

clean-index:  ## Remove index directory (DANGEROUS)
	rm -rf $(ARTIFACTS_DIR)/indexes/$(INDEX)

# -------------------------------------------------------------------
# Querying
# -------------------------------------------------------------------

ask:  ## Ask a question against the index
	$(PYTHON) scripts/ask.py \
		--index $(INDEX) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--q "$(QUERY)"

ask-dummy:  ## Ask using DummyEmbedder
	$(PYTHON) scripts/ask.py \
		--index $(INDEX) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--q "$(QUERY)" \
		--use-dummy-embeddings

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

results:  ## Launch results analyzer app
	$(PYTHON) -m streamlit run eval/app/results_analyzer.py

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------

tail-logs:  ## Tail JSONL query logs
	tail -n $(NUM_LOGS) $(ARTIFACTS_DIR)/logs/queries.jsonl | jq .

# -------------------------------------------------------------------
# Environment Utils
# -------------------------------------------------------------------

test:  ## Run tests with pytest
	$(PYTHON) -m pytest -q

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check .

fmt:  ## Run ruff formatter
	$(PYTHON) -m ruff format .

typecheck:  ## Run mypy type checks
	$(PYTHON) -m mypy rag

env-check:  ## Check Python environment
	@echo "PYTHON=$(PYTHON)"
	@$(PYTHON) -c "import sys; print(sys.executable)"
	@$(PYTHON) -c "import platform; print(platform.platform())"
