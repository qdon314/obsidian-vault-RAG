.PHONY: help index index-dummy ask ask-dummy results verdict tail-logs clean-index \
        index-regulatory index-regulatory-dummy normalize-regulatory \
        push-regulatory-s3 normalize-and-push-regulatory index-regulatory-push \
        test lint fmt typecheck env-check \
        docker-build docker-up docker-down image-push deploy-image \
        infra-init infra-plan infra-apply infra-destroy \
        ecs-up ecs-down ecs-status \
        ingest-remote eval-remote query-remote upload-eval-queries

# ---- Python interpreter (pin to .venv) ----
PYTHON := $(CURDIR)/.venv/bin/python

ifeq ($(wildcard $(PYTHON)),)
$(error Could not find PYTHON at $(PYTHON). Run: python3.11 -m venv .venv)
endif
# -----------------------------------------------------

ARTIFACTS_DIR ?= artifacts
INDEX ?= obsidian
CORPUS ?= /Users/quentindonnelly/Documents/Personal & Professional
QUERY ?= What are the applications of scaled dot-product attention?
NUM_LOGS ?= 20
REGULATORY_XML ?= data/ecfr/title-10-part-50.xml
REGULATORY_VERSION ?= 2025-02-01
REGULATORY_PART ?= 50
REGULATORY_S3_BUCKET ?=
REGULATORY_S3_PREFIX ?= regulatory/part-50
VERDICT_SCOPE ?=

help:  ## Show available commands
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
# Regulatory Corpus
# -------------------------------------------------------------------

index-regulatory:  ## Build regulatory index from eCFR XML
	$(PYTHON) scripts/ingest_regulatory.py \
		--xml-source "$(REGULATORY_XML)" \
		--part $(REGULATORY_PART) \
		--instrument-version "$(REGULATORY_VERSION)" \
		--source-revision "ecfr-$(REGULATORY_VERSION)" \
		--effective-date "$(REGULATORY_VERSION)" \
		--index-name regulatory \
		--artifacts-dir $(ARTIFACTS_DIR)

index-regulatory-dummy:  ## Build regulatory index with DummyEmbedder
	$(PYTHON) scripts/ingest_regulatory.py \
		--xml-source "$(REGULATORY_XML)" \
		--part $(REGULATORY_PART) \
		--instrument-version "$(REGULATORY_VERSION)" \
		--source-revision "ecfr-$(REGULATORY_VERSION)" \
		--effective-date "$(REGULATORY_VERSION)" \
		--index-name regulatory \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--use-dummy-embeddings

normalize-regulatory:  ## Normalize eCFR XML to canonical markdown (no indexing)
	$(PYTHON) scripts/regulatory_normalize.py \
		--xml-source "$(REGULATORY_XML)" \
		--part $(REGULATORY_PART) \
		--instrument-version "$(REGULATORY_VERSION)" \
		--source-revision "ecfr-$(REGULATORY_VERSION)" \
		--effective-date "$(REGULATORY_VERSION)"

push-regulatory-s3:  ## Push normalized regulatory part directory to S3
	@test -n "$(REGULATORY_S3_BUCKET)" || (echo "Set REGULATORY_S3_BUCKET=<bucket>"; exit 1)
	$(PYTHON) scripts/regulatory_push_s3.py \
		--bucket "$(REGULATORY_S3_BUCKET)" \
		--prefix "$(REGULATORY_S3_PREFIX)" \
		--part $(REGULATORY_PART)

normalize-and-push-regulatory:  ## Normalize regulatory corpus then push that part directory to S3
	$(MAKE) normalize-regulatory REGULATORY_PART=$(REGULATORY_PART)
	$(MAKE) push-regulatory-s3 REGULATORY_PART=$(REGULATORY_PART)

index-regulatory-push:  ## Build regulatory index, push normalized files to S3, wipe local normalized files
	@test -n "$(REGULATORY_S3_BUCKET)" || (echo "Set REGULATORY_S3_BUCKET=<bucket>"; exit 1)
	$(PYTHON) scripts/ingest_regulatory.py \
		--xml-source "$(REGULATORY_XML)" \
		--part $(REGULATORY_PART) \
		--instrument-version "$(REGULATORY_VERSION)" \
		--source-revision "ecfr-$(REGULATORY_VERSION)" \
		--effective-date "$(REGULATORY_VERSION)" \
		--index-name regulatory \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--push-s3-bucket "$(REGULATORY_S3_BUCKET)" \
		--push-s3-prefix "$(REGULATORY_S3_PREFIX)" \
		--wipe-local

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

verdict:  ## Generate eval verdict from latest run (requires baseline)
	# Produces eval/verdicts/verdict.md + verdict.json for human + CI consumption.
	$(PYTHON) eval/scripts/verdict.py \
		--current eval/runs/latest \
		--baseline eval/runs/baseline \
		--output eval/verdicts \
		$(if $(VERDICT_SCOPE),--scope $(VERDICT_SCOPE),)

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
	$(PYTHON) -m ruff check . --fix

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
AWS_REGION ?= us-east-1
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
ECR_REPO ?= obsidian-rag
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
ECR_IMAGE ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO):$(IMAGE_TAG)
IMAGE_PLATFORM ?= linux/amd64

docker-build:  ## Build Docker image locally
	docker buildx build --platform $(IMAGE_PLATFORM) -t rag-obsidian:dev --load .

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

image-push: docker-build  ## Build and push pinned image tag to ECR
	@test -n "$(AWS_ACCOUNT_ID)" || (echo "Set AWS_ACCOUNT_ID or configure AWS CLI credentials"; exit 1)
	@aws ecr describe-repositories --region $(AWS_REGION) --repository-names $(ECR_REPO) >/dev/null 2>&1 || \
		aws ecr create-repository --region $(AWS_REGION) --repository-name $(ECR_REPO) >/dev/null
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker buildx build --platform $(IMAGE_PLATFORM) -t $(ECR_IMAGE) --push .

deploy-image: image-push  ## Push pinned image and apply Terraform with app_image_tag
	@test -n "$$TF_VAR_openai_api_key" || (echo "Export TF_VAR_openai_api_key first"; exit 1)
	@test -n "$$TF_VAR_db_password" || (echo "Export TF_VAR_db_password first"; exit 1)
	cd infra && terraform apply -var "app_image_tag=$(IMAGE_TAG)"

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

# -------------------------------------------------------------------
# Remote Operations (ECS)
# -------------------------------------------------------------------

WORKERS ?= 3
CORPUS_ID ?= regulatory
INDEX_NAME ?= chunks_regulatory_v1
CORPUS_S3_PREFIX ?= regulatory
CORPUS_S3_BUCKET ?= $(shell cd infra && terraform output -raw corpus_bucket_name 2>/dev/null || echo "obsidian-rag-artifacts")
QUERY_SET ?= default
RUN_NAME ?=
EVAL_WORKERS ?= 1
USE_LLM_JUDGE ?= true
RUN_GENERATION ?= true

ingest-remote:  ## Run distributed ingestion on ECS (auto-scales workers)
	scripts/ecs_run_ingest.sh \
		--workers $(WORKERS) \
		--corpus-id $(CORPUS_ID) \
		--index-name $(INDEX_NAME) \
		$(if $(CORPUS_S3_PREFIX),--corpus-s3-prefix $(CORPUS_S3_PREFIX),) \
		$(if $(CORPUS_S3_BUCKET),--corpus-s3-bucket $(CORPUS_S3_BUCKET),)

eval-remote:  ## Run eval against remote backends on ECS
	scripts/ecs_run_eval.sh \
		--query-set $(QUERY_SET) \
		--max-workers $(EVAL_WORKERS) \
		$(if $(RUN_GENERATION),--run-generation,)
		$(if $(USE_LLM_JUDGE),--use-llm-judge,)
		$(if $(RUN_NAME),--run-name $(RUN_NAME),)

query-remote:  ## Run ad-hoc query on ECS
	scripts/ecs_run_query.sh "$(QUERY)"

upload-eval-queries:  ## Sync local eval datasets to S3
	@BUCKET=$$(cd infra && terraform output -raw corpus_bucket_name 2>/dev/null || echo "obsidian-rag-corpus"); \
	echo "Uploading eval datasets to s3://$$BUCKET/eval/queries/default/"; \
	aws s3 sync eval/datasets/ "s3://$$BUCKET/eval/queries/default/" --exclude "*.pyc"
