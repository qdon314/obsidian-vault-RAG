# Deployment Guide

## 1. Overview

This RAG system supports three deployment modes:

- **Local development** with Docker Compose (RAG app + Qdrant vector store)
- **Continuous integration** via GitHub Actions (lint, typecheck, test on every PR; Docker build + ECR push on merge to main)
- **Cloud deployment** via Terraform on AWS (ECS Fargate, ECR, S3, SSM Parameter Store)

The AWS deployment follows a **scale-to-zero** pattern: infrastructure is provisioned via Terraform and persists at near-zero cost, while ECS services are scaled up only when needed for demos or development.

---

## 2. Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker | 20.10+ | Container builds and local dev |
| Docker Compose | v2+ | Local multi-service stack |
| AWS CLI | v2 | AWS operations, ECR login, ECS scaling |
| Terraform | >= 1.5 | Infrastructure provisioning |
| GitHub repo | -- | CI/CD via GitHub Actions |

You also need:

- An **OpenAI API key** (for embeddings and generation)
- **AWS credentials** configured locally (`aws configure`) with permissions for ECR, ECS, S3, SSM, IAM, CloudWatch, Cloud Map, and EC2 (for subnet/VPC lookups)
- **GitHub Actions secrets** configured for the Docker build workflow (see [CI/CD](#5-cicd))

---

## 3. Local Development (Docker Compose)

### Quick start

1. Copy the example environment file and add your OpenAI API key:

   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-...
   ```

2. Mount your Obsidian vault (or any markdown corpus). Edit `docker-compose.yml` and uncomment the volume line under the `app` service:

   ```yaml
   volumes:
     - /path/to/your/vault:/data/vault:ro
   ```

3. Start Qdrant:

   ```bash
   docker compose up qdrant -d
   ```

   Qdrant exposes ports 6333 (HTTP) and 6334 (gRPC). The healthcheck ensures it is ready before the app starts.

4. Build an index from your corpus:

   ```bash
   docker compose run --rm app build-index --corpus /data/vault --index-name my_index
   ```

5. Query the index:

   ```bash
   docker compose run --rm app query --q "your question"
   ```

6. Tear down everything (including the Qdrant data volume):

   ```bash
   docker compose down -v
   ```

### How it works

- The app container is built from the multi-stage `Dockerfile` (Python 3.11 slim, with `openai` and `qdrant` extras installed).
- `docker-entrypoint.sh` dispatches commands: `build-index`, `query`, or `help`.
- `settings.docker.toml` is mounted as `settings.toml` inside the container, pre-configured to use Qdrant at `http://qdrant:6333`.
- The `artifacts/` directory is bind-mounted so index data persists on the host between container runs.
- Qdrant data is stored in a named Docker volume (`qdrant_data`).

### Available commands

```bash
# Show help
docker compose run --rm app help

# Build index
docker compose run --rm app build-index --corpus /data/vault --index-name my_index

# Query
docker compose run --rm app query --q "What is hexagonal architecture?"

# Run arbitrary commands inside the container
docker compose run --rm app bash
```

Note: keep local volume mounts portable in committed `docker-compose.yml` (template paths), and apply machine-specific mounts locally.

---

## 4. Configuration

### Configuration layers

The system uses a layered configuration approach:

1. **`settings.toml`** -- base config, baked into the Docker image at build time. Used for local development outside Docker.
2. **`settings.docker.toml`** -- Docker-specific overrides, mounted into the container by Docker Compose (replaces `settings.toml` inside the container).
3. **Environment variables** -- override any setting from either file. Used in ECS task definitions for AWS-specific values.

### Environment variable override convention

Environment variables use the pattern `RAG_<SECTION>__<KEY>=<value>` (note the **double underscore** separating section from key):

```bash
# Vector store configuration
RAG_VECTORSTORE__BACKEND=qdrant
RAG_VECTORSTORE__QDRANT_URL=http://qdrant:6333
RAG_VECTORSTORE__QDRANT_COLLECTION=obsidian

# Retrieval tuning
RAG_RETRIEVAL__TOP_K=12

# Disable reranking
RAG_RERANK__ENABLED=false

# LLM settings
RAG_LLM__MODEL=gpt-4.1-mini
RAG_LLM__TEMPERATURE=0.3
```

The double underscore is required because single underscores appear within field names (e.g., `qdrant_url`). Values are automatically coerced to match the type of the existing setting in the TOML file (booleans, integers, floats, and strings are supported).

### Key configuration sections

| Section | Key settings | Default |
|---------|-------------|---------|
| `[paths]` | `vault_dir`, `artifacts_dir`, `index_dir` | Local paths |
| `[chunking]` | `backend`, `chunk_size`, `overlap` | `obsidian_structural`, 800, 120 |
| `[embeddings]` | `backend`, `model` | `openai`, `text-embedding-3-large` |
| `[vectorstore]` | `backend`, `qdrant_url`, `qdrant_collection` | `jsonl` (local), `qdrant` (Docker) |
| `[retrieval]` | `top_k` | 8 |
| `[rerank]` | `enabled`, `backend`, `keep_k` | `true`, `heuristic`, 4 |
| `[llm]` | `backend`, `model`, `temperature` | `openai`, `gpt-4.1-mini`, 0.2 |

### Secrets

- **Local**: `OPENAI_API_KEY` in `.env` file (loaded by Docker Compose via `env_file`)
- **AWS**: Stored in SSM Parameter Store as a SecureString, injected into the ECS task definition as an environment variable by the Fargate agent. No application code changes required -- the OpenAI client reads `OPENAI_API_KEY` from the environment.

---

## 5. CI/CD

### CI workflow (`ci.yml`)

Runs on every push to `main` and every pull request targeting `main`.

Note: local development in this repository should use `make` / `./scripts/py` / `./scripts/pip`. The raw `pip`/`python` commands below describe CI runner internals only.

**Steps:**
1. Checkout code
2. Set up Python 3.11
3. Cache pip packages (keyed on `pyproject.toml` hash)
4. Install dependencies: `pip install -e ".[dev,openai,qdrant]"`
5. Lint: `ruff check .`
6. Format check: `ruff format --check .`
7. Type check: `mypy --config-file pyproject.toml src`
8. Test: `pytest -q`
9. Eval release gate:
   - run eval on curated queries
   - run `eval/scripts/verdict.py --fail-on-block`
   - upload `eval/verdicts/` artifacts

The lint/type/test steps run against in-memory/dummy backends and do not require external services.
The eval gate step does require `OPENAI_API_KEY`.

The eval gate requires a baseline run directory at `eval/runs/baseline/` in the repository.
See `docs/evaluation/verdict_release_gating.md` for baseline management and local commands.

### Docker workflow (`docker.yml`)

Runs on pushes to `main` only (not on PRs).

**Steps:**
1. Checkout code
2. Configure AWS credentials from GitHub secrets
3. Login to Amazon ECR
4. Build Docker image
5. Tag with both the git SHA and `latest`
6. Push both tags to ECR

The image is tagged as `<ecr-registry>/obsidian-rag:<git-sha>` and `<ecr-registry>/obsidian-rag:latest`.

### Required GitHub secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key with ECR push permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |
| `OPENAI_API_KEY` | Required by the eval gate job for generation + judge runs |

To set these, go to your GitHub repository **Settings > Secrets and variables > Actions** and add each secret.

### What is deliberately excluded from CI

- **No Terraform plan/apply in CI.** This is a cost-constrained portfolio project; AWS resources are managed manually to maintain control over when they exist and incur cost.
- **No integration tests against live OpenAI/Qdrant.** These would require API keys in CI and are better suited to a manually-triggered workflow.

---

## 6. AWS Deployment

### Infrastructure overview

```
                    GitHub Actions
                         |
                    docker.yml
                         |
                         v
                   +-----------+
                   |    ECR    |  Container registry
                   +-----------+  (obsidian-rag)
                         |
          +--------------+--------------+
          |                             |
          v                             v
  +---------------+            +---------------+
  |  ECS Fargate  |            |  ECS Fargate  |
  |   RAG App     |---DNS----->|    Qdrant     |
  | 0.25 vCPU     |            |  0.5 vCPU     |
  | 512 MB        |            |  1024 MB      |
  +---------------+            +---------------+
          |                             ^
          v                             |
  +---------------+           Cloud Map service
  |      S3       |           discovery (DNS)
  |   Artifacts   |           qdrant.obsidian-rag.local
  +---------------+

  +---------------+
  | SSM Parameter |  OPENAI_API_KEY
  |    Store      |  (SecureString)
  +---------------+
```

**Resources provisioned by Terraform:**

| Resource | Purpose | Module |
|----------|---------|--------|
| ECR repository | Container image registry with lifecycle policy (keeps 10 most recent images), scan-on-push enabled | `modules/ecr` |
| S3 bucket | Artifact storage (indexes, manifests). Versioned, encrypted (AES256), public access blocked, old versions expire after 30 days | `modules/s3` |
| SSM Parameter | OpenAI API key stored as SecureString | `modules/secrets` |
| ECS cluster | Fargate cluster hosting both services | `modules/ecs` |
| ECS service (Qdrant) | Qdrant vector store, registered with Cloud Map for DNS-based service discovery | `modules/ecs` |
| ECS service (App) | RAG application, configured via environment variables to connect to Qdrant at `qdrant.obsidian-rag.local:6333` | `modules/ecs` |
| Cloud Map namespace | Private DNS namespace (`obsidian-rag.local`) for inter-service communication | `modules/ecs` |
| CloudWatch log groups | Log streams for both app and Qdrant containers, 30-day retention | `modules/ecs` |
| IAM roles | Task execution role (ECR pull, SSM read) and task role (S3 read/write), both least-privilege scoped | `modules/ecs` |

### Distributed ingestion deployment notes

- Provision queue/database modules and wire outputs into ECS worker settings.
- Keep worker desired count at `0` by default; scale up when running ingestion jobs.
- Store ingestion DSN in secure parameter store and inject via `rds_dsn_arn`.
- See:
  - `docs/operations/distributed-ingestion.md` for runtime operations
  - `infra/modules/README.md` for module input/output wiring

### Deployment steps

1. **Navigate to the infrastructure directory and create your variables file:**

   ```bash
   cd infra
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit `terraform.tfvars`** with your actual values:

   ```hcl
   aws_region     = "us-east-1"
   project_name   = "obsidian-rag"
   openai_api_key = "sk-your-actual-key"

   # Find your default VPC subnet IDs:
   # aws ec2 describe-subnets --query 'Subnets[?DefaultForAz==`true`].SubnetId' --output text
   subnet_ids = ["subnet-abc123", "subnet-def456"]

# Optional: restrict network access
   # security_group_ids = ["sg-xxxxxxxx"]

   # Start scaled to zero
   app_desired_count    = 0
   qdrant_desired_count = 0
   ```

3. **Initialize and plan:**

   ```bash
   terraform init
   terraform plan
   ```

   Review the plan output. It should create ECR, S3, SSM, ECS cluster, task definitions, services, Cloud Map namespace, IAM roles, and CloudWatch log groups.

4. **Apply:**

   ```bash
   terraform apply
   ```

5. **Push the first Docker image** (or let the `docker.yml` CI workflow do this on the next merge to main):

   ```bash
   # Get the ECR repository URL from Terraform output
   ECR_URL=$(terraform output -raw ecr_repository_url)

   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR_URL"

   # Build and push from the project root
   cd ..
   docker build -t "$ECR_URL:latest" .
   docker push "$ECR_URL:latest"
   ```

6. **Scale up Qdrant first** (it must be healthy before the app can connect):

   ```bash
   aws ecs update-service \
     --cluster obsidian-rag \
     --service obsidian-rag-qdrant \
     --desired-count 1
   ```

   Wait for the service to reach a steady state:

   ```bash
   aws ecs wait services-stable \
     --cluster obsidian-rag \
     --services obsidian-rag-qdrant
   ```

7. **Scale up the app:**

   ```bash
   aws ecs update-service \
     --cluster obsidian-rag \
     --service obsidian-rag-app \
     --desired-count 1
   ```

### Scale to zero

When you are done, scale both services back to zero to stop incurring Fargate costs:

```bash
aws ecs update-service \
  --cluster obsidian-rag \
  --service obsidian-rag-app \
  --desired-count 0

aws ecs update-service \
  --cluster obsidian-rag \
  --service obsidian-rag-qdrant \
  --desired-count 0
```

### Cost estimate

| State | Monthly cost |
|-------|-------------|
| Scaled to zero (services at desired_count=0) | ~$0.10 (ECR storage only) |
| Running ~20 hours/month | ~$1 (Fargate compute + ECR + S3) |
| Running continuously | ~$15-25 (Fargate compute dominates) |

OpenAI API costs are separate and usage-dependent.

---

## 7. Architecture Notes

### Qdrant data is ephemeral

Qdrant runs on ECS Fargate without persistent storage (no EFS). This is a deliberate cost-saving choice. When the Qdrant task restarts, all data is lost. The intended workflow is:

1. Index artifacts (chunks, embeddings) are stored in S3 as versioned units.
2. On startup, the app rebuilds the Qdrant index from S3 artifacts.
3. Index data is small enough that this rebuild is fast.

For a production deployment requiring low-latency startup, add EFS with access points and mount targets to the Qdrant task definition.

### Service discovery

Cloud Map provides a private DNS namespace (`obsidian-rag.local`). The Qdrant service is registered as `qdrant.obsidian-rag.local`, and the app's ECS task definition is configured with:

```
RAG_VECTORSTORE__QDRANT_URL=http://qdrant.obsidian-rag.local:6333
```

No load balancer or public endpoint is needed for inter-service communication.

### Networking

Both services run in public subnets of the default VPC with `assign_public_ip = true` (required for Fargate tasks in public subnets to pull images from ECR). There is no ALB or public-facing endpoint -- the app is run as a batch task, not an HTTP service.

### Observability

- **CloudWatch Logs**: Both services log to CloudWatch under `/ecs/obsidian-rag/app` and `/ecs/obsidian-rag/qdrant`, with 30-day retention.
- **ECR image scanning**: Scan-on-push is enabled for vulnerability detection.

### IAM

Two IAM roles follow the principle of least privilege:

- **Task execution role** (`obsidian-rag-task-execution`): Allows ECS to pull images from ECR and read the OpenAI API key from SSM Parameter Store.
- **Task role** (`obsidian-rag-task`): Allows the running container to read from and write to the S3 artifacts bucket.

### Terraform state

Terraform state is stored locally by default. The `main.tf` file includes a commented-out S3 backend configuration for remote state if needed:

```hcl
backend "s3" {
  bucket = "obsidian-rag-tfstate"
  key    = "infra/terraform.tfstate"
  region = "us-east-1"
}
```

---

## 8. Troubleshooting

### Qdrant not ready when app starts

**Symptom:** App fails to connect to Qdrant on startup.

**Local (Docker Compose):** The `depends_on` with `condition: service_healthy` ensures Qdrant is ready before the app starts. If you see connection errors, check that the Qdrant healthcheck is passing:

```bash
docker compose ps
docker compose logs qdrant
```

**AWS (ECS):** There is no built-in dependency ordering between ECS services. Scale up Qdrant first and wait for it to stabilize before scaling the app:

```bash
aws ecs update-service --cluster obsidian-rag --service obsidian-rag-qdrant --desired-count 1
aws ecs wait services-stable --cluster obsidian-rag --services obsidian-rag-qdrant
# Then scale up the app
```

### ECR login failures

**Symptom:** `docker push` returns authorization errors.

```bash
# Re-authenticate (tokens expire after 12 hours)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

Make sure your AWS CLI credentials have the `ecr:GetAuthorizationToken` permission.

### Missing environment variables

**Symptom:** App starts but fails at runtime with missing config errors.

- **Local:** Verify your `.env` file exists and contains `OPENAI_API_KEY`. Docker Compose loads it via the `env_file` directive.
- **AWS:** Check that the SSM parameter exists and the task execution role has permission to read it:

  ```bash
  aws ssm get-parameter --name /obsidian-rag/openai-api-key --with-decryption
  ```

### ECS task keeps stopping

**Symptom:** Desired count is 1 but running count stays at 0.

Check the stopped task's reason:

```bash
# List stopped tasks
aws ecs list-tasks --cluster obsidian-rag --desired-status STOPPED

# Describe the most recent stopped task
aws ecs describe-tasks --cluster obsidian-rag --tasks <task-arn>
```

Common causes:
- Image not found in ECR (push an image first)
- Insufficient permissions (IAM role misconfigured)
- Container exits immediately (check CloudWatch logs)

### Viewing container logs

```bash
# Local
docker compose logs app
docker compose logs qdrant

# AWS (CloudWatch)
aws logs tail /ecs/obsidian-rag/app --follow
aws logs tail /ecs/obsidian-rag/qdrant --follow
```

### Terraform state issues

If Terraform state gets out of sync with actual AWS resources:

```bash
# Refresh state from AWS
terraform refresh

# Import an existing resource
terraform import module.ecr.aws_ecr_repository.this obsidian-rag
```

### Subnet ID lookup

If you need to find your default VPC subnet IDs for `terraform.tfvars`:

```bash
aws ec2 describe-subnets \
  --query 'Subnets[?DefaultForAz==`true`].[SubnetId,AvailabilityZone]' \
  --output table
```
