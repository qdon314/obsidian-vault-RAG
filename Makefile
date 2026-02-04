.PHONY: help index index-dummy ask ask-dummy results tail-logs clean-index \
        test lint fmt typecheck env-check \
        docker-build docker-up docker-down \
        infra-init infra-plan infra-apply infra-destroy \
        ecs-up ecs-down ecs-status

# ---- Python interpreter (pin to .venv) ----
PYTHON := $(CURDIR)/.venv/bin/python

ifeq ($(wildcard $(PYTHON)),)
$(error Could not find PYTHON at $(PYTHON). Run: python3.11 -m venv .venv)
endif
# -----------------------------------------------------

ARTIFACTS_DIR ?= artifacts
INDEX ?= obsidian_proposition_index
CORPUS ?= /Users/quentindonnelly/Documents/Personal & Professional
QUERY ?= What are the applications of scaled dot-product attention?
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
	$(PYTHON) -m mypy --config-file pyproject.toml src

env-check:  ## Check Python environment
	@echo "PYTHON=$(PYTHON)"
	@$(PYTHON) -c "import sys; print(sys.executable)"
	@$(PYTHON) -c "import platform; print(platform.platform())"

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

ECS_CLUSTER ?= obsidian-rag
ECS_APP_SERVICE ?= $(ECS_CLUSTER)-app
ECS_QDRANT_SERVICE ?= $(ECS_CLUSTER)-qdrant

docker-build:  ## Build Docker image locally
	docker build -t rag-obsidian:dev .

docker-index:  ## Build index inside Docker (starts Qdrant + runs indexer)
	docker compose run --rm app build-index \
		--corpus /data/vault \
		--index-name $(INDEX) \
		--artifacts-dir /app/artifacts

docker-query: ## Query index inside Docker (starts Qdrant + runs asker)
	docker compose run --rm app query \
		--index $(INDEX) \
		--artifacts-dir /app/artifacts \
		--q "$(QUERY)"

docker-up:  ## Start local stack (Qdrant + app shell)
	docker compose up qdrant -d

docker-down:  ## Tear down local stack and volumes
	docker compose down -v

# -------------------------------------------------------------------
# Terraform
# -------------------------------------------------------------------

infra-init:  ## Initialize Terraform
	cd infra && terraform init

infra-plan:  ## Plan Terraform changes
	cd infra && terraform plan

infra-apply:  ## Apply Terraform changes
	cd infra && terraform apply

infra-destroy:  ## Destroy all Terraform-managed infrastructure
	cd infra && terraform destroy

# -------------------------------------------------------------------
# ECS Scaling
# -------------------------------------------------------------------

ecs-up:  ## Scale Qdrant then app to 1 task each
	aws ecs update-service --cluster $(ECS_CLUSTER) \
		--service $(ECS_QDRANT_SERVICE) --desired-count 1 \
		--no-cli-pager
	@echo "Waiting for Qdrant to stabilize..."
	aws ecs wait services-stable --cluster $(ECS_CLUSTER) \
		--services $(ECS_QDRANT_SERVICE)
	aws ecs update-service --cluster $(ECS_CLUSTER) \
		--service $(ECS_APP_SERVICE) --desired-count 1 \
		--force-new-deployment --no-cli-pager

ecs-down:  ## Scale app then Qdrant to 0 tasks
	aws ecs update-service --cluster $(ECS_CLUSTER) \
		--service $(ECS_APP_SERVICE) --desired-count 0 \
		--no-cli-pager
	aws ecs update-service --cluster $(ECS_CLUSTER) \
		--service $(ECS_QDRANT_SERVICE) --desired-count 0 \
		--no-cli-pager

ecs-status:  ## Show running ECS task counts
	@aws ecs describe-services --cluster $(ECS_CLUSTER) \
		--services $(ECS_APP_SERVICE) $(ECS_QDRANT_SERVICE) \
		--query 'services[].{name:serviceName,running:runningCount,desired:desiredCount}' \
		--output table --no-cli-pager
