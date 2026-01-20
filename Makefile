.PHONY: help index ask ask-dummy tail-logs clean-index eval

ARTIFACTS_DIR ?= artifacts
INDEX ?= obsidian
CORPUS ?= "/Users/quentindonnelly/Documents/Personal & Professional"
QUERY ?= "What is this project about?"
NUM_LOGS ?= 20

help:  ## Show available commands
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------

index:  ## Build index from corpus (JSONL store)
	python scripts/build_index.py \
		--index-name $(INDEX) \
		--corpus $(CORPUS) \
		--artifacts-dir $(ARTIFACTS_DIR)

index-dummy:  ## Build index using DummyEmbedder
	python scripts/build_index.py \
		--index-name $(INDEX) \
		--corpus $(CORPUS) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--use-dummy-embeddings

clean-index:  ## Remove index directory (DANGEROUS)
	rm -rf $(ARTIFACTS_DIR)/indexes/$(INDEX)

# -------------------------------------------------------------------
# Querying
# -------------------------------------------------------------------

ask:  ## Ask a question against the index
	python scripts/ask.py \
		--index $(INDEX) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--q $(QUERY)

ask-dummy:  ## Ask using DummyEmbedder
	python scripts/ask.py \
		--index $(INDEX) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--q $(QUERY) \
		--use-dummy-embeddings

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

results:  ## Launch results analyzer app
	python -m streamlit run eval/app/results_analyzer.py

# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------

tail-logs:  ## Tail JSONL query logs
	tail -n 20 $(ARTIFACTS_DIR)/logs/queries.jsonl | jq .

tail-logs:  ## Tail JSONL query logs
	tail -n $(NUM_LOGS) $(ARTIFACTS_DIR)/logs/queries.jsonl | jq .

