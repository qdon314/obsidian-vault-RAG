# Phase 4: ECS/Fargate Deployment — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ingest-orchestrator and query-eval ECS task definitions, launcher scripts, and Make targets to complete the Phase 4 deployment story.

**Architecture:** Two new run-to-completion ECS task definitions (no services) that reuse the existing Docker image. An orchestrator script unifies enumerate→poll→finalize. A remote eval script downloads queries from S3, runs the harness, and uploads results. Shell launcher scripts wrap `aws ecs run-task` with auto-scaling.

**Tech Stack:** Terraform (HCL), Python 3.11, Bash, AWS CLI (ECS, S3, CloudWatch Logs)

**Design doc:** `docs/plans/2026-02-13-phase4-ecs-deployment-design.md`

---

### Task 1: Terraform — ECS module variables

Add new variables for the orchestrator and query-eval task definitions.

**Files:**
- Modify: `infra/modules/ecs/variables.tf` (append after line 145)

**Step 1: Add variables**

Append these variable blocks to the end of `infra/modules/ecs/variables.tf`:

```hcl
# Orchestrator task sizing
variable "orchestrator_cpu" {
  description = "CPU units for ingest orchestrator task"
  type        = number
  default     = 256
}

variable "orchestrator_memory" {
  description = "Memory (MB) for ingest orchestrator task"
  type        = number
  default     = 512
}

# Query/eval task sizing
variable "query_eval_cpu" {
  description = "CPU units for query/eval task"
  type        = number
  default     = 256
}

variable "query_eval_memory" {
  description = "Memory (MB) for query/eval task"
  type        = number
  default     = 512
}

# S3 prefixes for eval and manifests
variable "eval_s3_prefix" {
  description = "Prefix for eval queries and run results in S3"
  type        = string
  default     = "eval"
}

variable "manifests_s3_prefix" {
  description = "Prefix for index manifest objects in S3"
  type        = string
  default     = "manifests"
}
```

**Step 2: Commit**

```
git add infra/modules/ecs/variables.tf
git commit -m "infra: add orchestrator and query-eval ECS variables"
```

---

### Task 2: Terraform — ECS module task definitions and log groups

Add the two new task definitions (no services) and their CloudWatch log groups.

**Files:**
- Modify: `infra/modules/ecs/main.tf` (append after line 228)

**Step 1: Add orchestrator log group + task definition**

Append to `infra/modules/ecs/main.tf`:

```hcl
# --- Ingest orchestrator (run-to-completion) ---
resource "aws_cloudwatch_log_group" "orchestrator" {
  name              = "/ecs/${var.cluster_name}/ingest-orchestrator"
  retention_in_days = 30
  tags              = var.tags
}

resource "aws_ecs_task_definition" "ingest_orchestrator" {
  family                   = "${var.cluster_name}-ingest-orchestrator"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.orchestrator_cpu
  memory                   = var.orchestrator_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "ingest-orchestrator"
      image     = var.app_image
      essential = true
      command   = ["python", "scripts/run_orchestrator.py"]
      environment = [
        { name = "RAG_VECTORSTORE__BACKEND", value = "qdrant" },
        { name = "RAG_VECTORSTORE__QDRANT_URL", value = "http://qdrant.${var.cluster_name}.local:6333" },
        { name = "RAG_DISTRIBUTED_INGESTION__ENABLED", value = "true" },
        { name = "RAG_DISTRIBUTED_INGESTION__SQS_QUEUE_URL", value = var.sqs_queue_url },
        { name = "RAG_DISTRIBUTED_INGESTION__CORPUS_S3_BUCKET", value = var.s3_bucket_name },
        { name = "RAG_DISTRIBUTED_INGESTION__CORPUS_S3_PREFIX", value = var.corpus_s3_prefix },
        { name = "RAG_CHUNK_STORAGE__BACKEND", value = "s3" },
        { name = "RAG_CHUNK_STORAGE__S3_BUCKET", value = var.s3_bucket_name },
        { name = "RAG_CHUNK_STORAGE__S3_PREFIX", value = var.chunk_s3_prefix },
        { name = "RAG_MANIFESTS_S3_PREFIX", value = var.manifests_s3_prefix },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = var.openai_api_key_arn },
        { name = "RAG_DISTRIBUTED_INGESTION__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
        { name = "RAG_CHUNK_STORAGE__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.orchestrator.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "orchestrator"
        }
      }
    }
  ])

  tags = var.tags
}
```

**Step 2: Add query-eval log group + task definition**

Continue appending to `infra/modules/ecs/main.tf`:

```hcl
# --- Query/eval task (run-to-completion) ---
resource "aws_cloudwatch_log_group" "query_eval" {
  name              = "/ecs/${var.cluster_name}/query-eval"
  retention_in_days = 30
  tags              = var.tags
}

resource "aws_ecs_task_definition" "query_eval" {
  family                   = "${var.cluster_name}-query-eval"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.query_eval_cpu
  memory                   = var.query_eval_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "query-eval"
      image     = var.app_image
      essential = true
      command   = ["python", "scripts/run_remote_eval.py"]
      environment = [
        { name = "RAG_VECTORSTORE__BACKEND", value = "qdrant" },
        { name = "RAG_VECTORSTORE__QDRANT_URL", value = "http://qdrant.${var.cluster_name}.local:6333" },
        { name = "RAG_CHUNK_STORAGE__BACKEND", value = "s3" },
        { name = "RAG_CHUNK_STORAGE__S3_BUCKET", value = var.s3_bucket_name },
        { name = "RAG_CHUNK_STORAGE__S3_PREFIX", value = var.chunk_s3_prefix },
        { name = "RAG_EVAL_S3_PREFIX", value = var.eval_s3_prefix },
        { name = "RAG_MANIFESTS_S3_PREFIX", value = var.manifests_s3_prefix },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = var.openai_api_key_arn },
        { name = "RAG_DISTRIBUTED_INGESTION__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
        { name = "RAG_CHUNK_STORAGE__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.query_eval.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "query-eval"
        }
      }
    }
  ])

  tags = var.tags
}
```

**Step 3: Commit**

```
git add infra/modules/ecs/main.tf
git commit -m "infra: add orchestrator and query-eval ECS task definitions"
```

---

### Task 3: Terraform — ECS module outputs

Expose the new task definition ARNs so launcher scripts can reference them.

**Files:**
- Modify: `infra/modules/ecs/outputs.tf` (append after line 24)

**Step 1: Add outputs**

Append to `infra/modules/ecs/outputs.tf`:

```hcl
output "orchestrator_task_definition_arn" {
  description = "ARN of the ingest orchestrator task definition"
  value       = aws_ecs_task_definition.ingest_orchestrator.arn
}

output "query_eval_task_definition_arn" {
  description = "ARN of the query/eval task definition"
  value       = aws_ecs_task_definition.query_eval.arn
}

output "worker_service_name" {
  description = "Name of the ingest worker ECS service"
  value       = aws_ecs_service.ingest_worker.name
}
```

**Step 2: Commit**

```
git add infra/modules/ecs/outputs.tf
git commit -m "infra: expose orchestrator and query-eval task definition ARNs"
```

---

### Task 4: Terraform — Root module wiring

Wire the new ECS module variables and outputs through the root module.

**Files:**
- Modify: `infra/main.tf` (add params to ecs module call, lines ~81-105)
- Modify: `infra/variables.tf` (append new root-level variables)
- Modify: `infra/outputs.tf` (append new outputs)
- Modify: `infra/terraform.tfvars.example` (append new example values)

**Step 1: Add new variables to root `infra/variables.tf`**

Append to end of file:

```hcl
variable "orchestrator_cpu" {
  description = "CPU units for ingest orchestrator task"
  type        = number
  default     = 256
}

variable "orchestrator_memory" {
  description = "Memory (MB) for ingest orchestrator task"
  type        = number
  default     = 512
}

variable "query_eval_cpu" {
  description = "CPU units for query/eval task"
  type        = number
  default     = 256
}

variable "query_eval_memory" {
  description = "Memory (MB) for query/eval task"
  type        = number
  default     = 512
}

variable "eval_s3_prefix" {
  description = "Prefix for eval queries and run results in S3"
  type        = string
  default     = "eval"
}

variable "manifests_s3_prefix" {
  description = "Prefix for index manifest objects in S3"
  type        = string
  default     = "manifests"
}
```

**Step 2: Wire into ECS module call in `infra/main.tf`**

Add these parameters to the `module "ecs"` block (after `chunk_max_s3_workers`
on line 102):

```hcl
  orchestrator_cpu     = var.orchestrator_cpu
  orchestrator_memory  = var.orchestrator_memory
  query_eval_cpu       = var.query_eval_cpu
  query_eval_memory    = var.query_eval_memory
  eval_s3_prefix       = var.eval_s3_prefix
  manifests_s3_prefix  = var.manifests_s3_prefix
```

**Step 3: Add root outputs in `infra/outputs.tf`**

Append to end of file:

```hcl
output "orchestrator_task_definition_arn" {
  description = "ARN of the ingest orchestrator task definition"
  value       = module.ecs.orchestrator_task_definition_arn
}

output "query_eval_task_definition_arn" {
  description = "ARN of the query/eval task definition"
  value       = module.ecs.query_eval_task_definition_arn
}

output "worker_service_name" {
  description = "Name of the ingest worker ECS service"
  value       = module.ecs.worker_service_name
}
```

**Step 4: Update `infra/terraform.tfvars.example`**

Append:

```hcl

# Orchestrator sizing
orchestrator_cpu    = 256
orchestrator_memory = 512

# Query/eval sizing
query_eval_cpu    = 256
query_eval_memory = 512

# S3 prefixes
eval_s3_prefix      = "eval"
manifests_s3_prefix = "manifests"
```

**Step 5: Validate with `terraform plan`**

Run: `cd infra && terraform init && terraform validate`
Expected: "Success! The configuration is valid."

**Step 6: Commit**

```
git add infra/main.tf infra/variables.tf infra/outputs.tf infra/terraform.tfvars.example
git commit -m "infra: wire orchestrator and query-eval through root module"
```

---

### Task 5: Python — Orchestrator polling helper (with tests)

Extract a testable polling function before writing the full orchestrator script.
This is the only non-trivial logic in the orchestrator — the rest is composition
of existing `start_ingestion.py` and `finalize_job.py` code.

**Files:**
- Create: `src/rag/app/ingestion/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write the failing test**

Create `tests/test_orchestrator.py`:

```python
"""Tests for the orchestrator poll loop."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from rag.app.ingestion.orchestrator import poll_until_complete
from rag.domain.ingestion import JobStatus, TaskStatus


def _mock_job_store(counts_sequence: list[dict[TaskStatus, int]]) -> MagicMock:
    """Return a mock job store that returns successive task count dicts."""
    store = MagicMock()
    store.get_task_counts = MagicMock(side_effect=counts_sequence)
    return store


def test_poll_completes_when_all_succeeded():
    job_id = uuid.uuid4()
    store = _mock_job_store([
        {TaskStatus.PENDING: 5, TaskStatus.RUNNING: 5},
        {TaskStatus.SUCCEEDED: 8, TaskStatus.RUNNING: 2},
        {TaskStatus.SUCCEEDED: 10},
    ])

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=10,
    )

    assert result.all_succeeded is True
    assert result.succeeded == 10
    assert result.failed == 0


def test_poll_detects_failures():
    job_id = uuid.uuid4()
    store = _mock_job_store([
        {TaskStatus.SUCCEEDED: 8, TaskStatus.FAILED: 2},
    ])

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=10,
    )

    assert result.all_succeeded is False
    assert result.succeeded == 8
    assert result.failed == 2


def test_poll_times_out():
    job_id = uuid.uuid4()
    # Always returns tasks in progress — will never complete
    store = _mock_job_store(
        [{TaskStatus.RUNNING: 10}] * 100,
    )

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=0,  # immediate timeout
    )

    assert result.timed_out is True
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/test_orchestrator.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError" (module doesn't exist yet)

**Step 3: Write the implementation**

Create `src/rag/app/ingestion/orchestrator.py`:

```python
"""Orchestrator: poll loop for monitoring distributed ingestion progress.

The poll loop queries the Postgres job store for task counts until all tasks
have reached a terminal state (SUCCEEDED or FAILED) or a timeout is reached.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.domain.ingestion import TaskStatus

if TYPE_CHECKING:
    from rag.ports import IngestJobStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PollResult:
    """Outcome of polling for task completion."""

    succeeded: int
    failed: int
    pending: int
    running: int
    retryable: int
    timed_out: bool

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.pending + self.running + self.retryable

    @property
    def all_succeeded(self) -> bool:
        return not self.timed_out and self.failed == 0 and self.pending == 0 and self.running == 0 and self.retryable == 0


def poll_until_complete(
    *,
    job_id: uuid.UUID,
    job_store: IngestJobStore,
    total_tasks: int,
    poll_interval_s: float = 30.0,
    timeout_s: float = 7200.0,
) -> PollResult:
    """Poll task counts until all tasks reach a terminal state or timeout.

    Terminal states: SUCCEEDED, FAILED (past max retries).
    Non-terminal states: PENDING, RUNNING, RETRYABLE.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        counts = job_store.get_task_counts(job_id)
        succeeded = counts.get(TaskStatus.SUCCEEDED, 0)
        failed = counts.get(TaskStatus.FAILED, 0)
        pending = counts.get(TaskStatus.PENDING, 0)
        running = counts.get(TaskStatus.RUNNING, 0)
        retryable = counts.get(TaskStatus.RETRYABLE, 0)

        logger.info(
            "Poll: %d/%d succeeded, %d failed, %d pending, %d running, %d retryable",
            succeeded, total_tasks, failed, pending, running, retryable,
        )

        # Done if no tasks are still in-flight
        in_flight = pending + running + retryable
        if in_flight == 0:
            return PollResult(
                succeeded=succeeded,
                failed=failed,
                pending=0,
                running=0,
                retryable=0,
                timed_out=False,
            )

        if time.monotonic() >= deadline:
            logger.warning("Timeout reached with %d tasks still in-flight", in_flight)
            return PollResult(
                succeeded=succeeded,
                failed=failed,
                pending=pending,
                running=running,
                retryable=retryable,
                timed_out=True,
            )

        time.sleep(poll_interval_s)
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/test_orchestrator.py -v`
Expected: 3 passed

**Step 5: Commit**

```
git add src/rag/app/ingestion/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator poll loop with tests"
```

---

### Task 6: Python — `run_orchestrator.py` script

Combines enumerate → poll → finalize into a single entry point. Reuses
existing code from `start_ingestion.py` (enumeration) and `finalize_job.py`
(finalization), with the new poll loop in between.

**Files:**
- Create: `scripts/run_orchestrator.py`

**Step 1: Write the script**

Create `scripts/run_orchestrator.py`:

```python
"""CLI: Run the full distributed ingestion lifecycle.

Combines: enumerate docs → poll for worker completion → finalize job.
Designed to run as a single ECS task (ingest-orchestrator).

Usage:
    ./scripts/py scripts/run_orchestrator.py \
        --corpus /path/to/vault \
        --corpus-id regulations_v1 \
        --index-name regulatory
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3  # type: ignore[import-untyped]
import psycopg2  # type: ignore[import-untyped]
from dotenv import load_dotenv

from rag import settings
from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.adapters.queue.sqs_task_queue import SQSTaskQueue
from rag.app.container import build_container
from rag.app.ingestion.enumerator import Enumerator
from rag.app.ingestion.orchestrator import poll_until_complete
from rag.domain.index_manifest import IndexManifest
from rag.domain.ingestion import JobStatus, TaskStatus

log = logging.getLogger("orchestrator")


def _count_chunks(dsn: str) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunk_index")
            return cur.fetchone()[0]  # type: ignore[index]
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full ingestion lifecycle.")
    ap.add_argument("--corpus", required=True, help="Path to corpus directory.")
    ap.add_argument("--corpus-id", required=True, help="Unique corpus identifier.")
    ap.add_argument("--index-name", required=True, help="Index name for manifest.")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs (0=all).")
    ap.add_argument("--qdrant-collection", type=str, default=None)
    ap.add_argument("--poll-interval", type=float, default=30.0, help="Poll interval in seconds.")
    ap.add_argument("--timeout", type=float, default=7200.0, help="Max wait time in seconds.")
    ap.add_argument("--force-finalize", action="store_true", help="Finalize even if tasks failed.")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[orchestrator] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true")
        raise SystemExit(1)
    if cfg.distributed_ingestion.postgres_dsn is None:
        log.error("distributed_ingestion.postgres_dsn must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.corpus_s3_bucket is None:
        log.error("distributed_ingestion.corpus_s3_bucket must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.sqs_queue_url is None:
        log.error("distributed_ingestion.sqs_queue_url must be set")
        raise SystemExit(1)

    # ── Phase 1: Enumerate ─────────────────────────────────────────
    log.info("Phase 1: Enumerating documents...")

    container = build_container()
    vault_root = Path(args.corpus).expanduser().resolve()
    docs, _report = container.ingestor.ingest([str(vault_root)])

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    log.info("Ingested %d docs, now creating job...", len(docs))

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    job_store.ensure_schema()

    raw_store = S3RawDocumentStore(
        bucket=cfg.distributed_ingestion.corpus_s3_bucket,
        prefix=f"{cfg.distributed_ingestion.corpus_s3_prefix}/{args.corpus_id}/raw",
    )
    queue = SQSTaskQueue(queue_url=cfg.distributed_ingestion.sqs_queue_url)

    enumerator = Enumerator(
        job_store=job_store,
        raw_document_store=raw_store,
        task_queue=queue,
    )

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    index_id = f"{args.index_name}_{cfg.chunking.backend}_{cfg.embeddings.model}_{ts}"

    job = enumerator.enumerate(
        docs=docs,
        corpus_id=args.corpus_id,
        index_id=index_id,
        chunking_strategy=cfg.chunking.backend,
        embedder_model=cfg.embeddings.model,
        qdrant_collection=args.qdrant_collection or cfg.vectorstore.qdrant_collection,
    )

    log.info("Job %s created (status=%s, tasks=%d)", job.job_id, job.status.value, len(docs))

    if len(docs) == 0:
        log.info("No documents to process. Done.")
        return

    # ── Phase 2: Poll ──────────────────────────────────────────────
    log.info("Phase 2: Polling for task completion...")

    result = poll_until_complete(
        job_id=job.job_id,
        job_store=job_store,
        total_tasks=len(docs),
        poll_interval_s=args.poll_interval,
        timeout_s=args.timeout,
    )

    if result.timed_out:
        log.error(
            "Timed out: %d succeeded, %d failed, %d still in-flight",
            result.succeeded, result.failed,
            result.pending + result.running + result.retryable,
        )
        job_store.update_job_status(job.job_id, JobStatus.FAILED, stats={
            "reason": "timeout",
            "succeeded": result.succeeded,
            "failed": result.failed,
        })
        raise SystemExit(1)

    log.info("All tasks terminal: %d succeeded, %d failed", result.succeeded, result.failed)

    if result.failed > 0 and not args.force_finalize:
        log.error(
            "%d tasks failed. Use --force-finalize to proceed anyway.", result.failed,
        )
        job_store.update_job_status(job.job_id, JobStatus.FAILED, stats={
            "succeeded": result.succeeded,
            "failed": result.failed,
        })
        raise SystemExit(1)

    # ── Phase 3: Finalize ──────────────────────────────────────────
    log.info("Phase 3: Finalizing job...")

    chunk_count = _count_chunks(cfg.distributed_ingestion.postgres_dsn)
    log.info("Chunk index: %d chunks", chunk_count)

    manifest = IndexManifest.create(
        index_name=args.index_name,
        corpus=args.corpus_id,
        doc_count=result.succeeded,
        chunk_count=chunk_count,
        chunking={"backend": cfg.chunking.backend},
        embedding={"model": cfg.embeddings.model},
        store={
            "type": "s3+qdrant",
            "bucket": cfg.distributed_ingestion.corpus_s3_bucket,
            "collection": args.qdrant_collection or cfg.vectorstore.qdrant_collection,
        },
    )

    bucket = cfg.distributed_ingestion.corpus_s3_bucket
    prefix = cfg.distributed_ingestion.corpus_s3_prefix or ""
    s3_key = (
        f"{prefix}/manifests/{index_id}/manifest.json"
        if prefix
        else f"manifests/{index_id}/manifest.json"
    )

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("Uploaded manifest to s3://%s/%s", bucket, s3_key)

    job_store.update_job_status(
        job.job_id,
        JobStatus.COMPLETED,
        stats={
            "doc_count": result.succeeded,
            "chunk_count": chunk_count,
            "failed_tasks": result.failed,
            "manifest_s3_key": s3_key,
        },
    )
    log.info("Job %s marked COMPLETED", job.job_id)


if __name__ == "__main__":
    main()
```

**Step 2: Verify lint passes**

Run: `./scripts/py -m ruff check scripts/run_orchestrator.py`
Expected: no errors (or fix any import issues)

**Step 3: Commit**

```
git add scripts/run_orchestrator.py
git commit -m "feat: add run_orchestrator.py combining enumerate/poll/finalize"
```

---

### Task 7: Python — `run_remote_eval.py` script

Downloads eval queries from S3, builds a remote-backend container, runs the
eval harness, uploads results to S3.

**Files:**
- Create: `scripts/run_remote_eval.py`

**Step 1: Write the script**

Create `scripts/run_remote_eval.py`:

```python
"""CLI: Run evaluation against remote backends (Qdrant + S3 chunk store).

Downloads eval queries from S3, runs the harness, uploads results to S3.
Designed to run as an ECS task (query-eval).

Usage:
    ./scripts/py scripts/run_remote_eval.py \
        --query-set default \
        --run-name my-eval-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import boto3  # type: ignore[import-untyped]
from dotenv import load_dotenv

from rag.app.container import build_container
from rag.eval.harness import load_eval_queries, run_full_eval, save_run
from rag.settings import load_settings

log = logging.getLogger("remote-eval")


def _download_s3_prefix(bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects under an S3 prefix to a local directory."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            downloaded.append(local_path)
            log.info("Downloaded s3://%s/%s -> %s", bucket, key, local_path)

    return downloaded


def _upload_directory(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload all files in a local directory to S3."""
    s3 = boto3.client("s3")
    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir)}"
            s3.upload_file(str(path), bucket, key)
            log.info("Uploaded %s -> s3://%s/%s", path.name, bucket, key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run eval against remote backends.")
    ap.add_argument("--query-set", default="default", help="Name of query set in S3.")
    ap.add_argument("--run-name", default=None, help="Optional run name label.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--keep-k", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=1500)
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--use-llm-judge", action="store_true")
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--score-ids", choices=("retrieved", "reranked"), default="reranked")
    ap.add_argument("--manifest", type=str, default=None, help="Manifest URI (local or s3://).")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[remote-eval] %(message)s", level=logging.INFO)

    cfg = load_settings()

    # Determine S3 bucket and prefixes
    bucket = cfg.distributed_ingestion.corpus_s3_bucket or cfg.chunk_storage.s3_bucket
    if not bucket:
        log.error("No S3 bucket configured (need distributed_ingestion.corpus_s3_bucket or chunk_storage.s3_bucket)")
        raise SystemExit(1)

    eval_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")

    # ── Download eval queries from S3 ──────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        queries_dir = Path(tmpdir) / "queries"
        queries_dir.mkdir()

        s3_queries_prefix = f"{eval_prefix}/queries/{args.query_set}"
        log.info("Downloading eval queries from s3://%s/%s", bucket, s3_queries_prefix)
        downloaded = _download_s3_prefix(bucket, s3_queries_prefix, queries_dir)

        if not downloaded:
            log.error("No eval queries found at s3://%s/%s", bucket, s3_queries_prefix)
            raise SystemExit(1)

        # Find the queries JSONL file
        jsonl_files = [f for f in downloaded if f.suffix == ".jsonl"]
        if not jsonl_files:
            log.error("No .jsonl file found in downloaded queries")
            raise SystemExit(1)

        queries_path = jsonl_files[0]
        log.info("Using queries file: %s", queries_path)
        eval_queries = load_eval_queries(queries_path)
        log.info("Loaded %d eval queries", len(eval_queries))

        # ── Build container with remote backends ───────────────────
        container = build_container(cfg=cfg)

        # ── LLM judge setup ────────────────────────────────────────
        judge_client = None
        if args.use_llm_judge:
            from openai import OpenAI
            api_key = cfg.secrets.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key required for LLM judge")
            judge_client = OpenAI(api_key=api_key)

        # ── Load manifest if provided ──────────────────────────────
        manifest = None
        if args.manifest:
            from rag.domain.index_manifest import IndexManifest
            manifest = IndexManifest.load_uri(args.manifest)

        # ── Run eval ───────────────────────────────────────────────
        log.info("Running evaluation...")
        run = run_full_eval(
            eval_queries=eval_queries,
            container=container,
            queries_path=str(queries_path),
            manifest=manifest,
            top_k=args.top_k,
            keep_k=args.keep_k,
            token_budget=args.token_budget,
            run_generation=args.run_generation,
            use_llm_judge=args.use_llm_judge,
            judge_client=judge_client,
            judge_model=args.judge_model if args.use_llm_judge else None,
            score_ids=args.score_ids,
            run_name=args.run_name,
        )

        # ── Save locally then upload to S3 ─────────────────────────
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        run_label = args.run_name or timestamp
        local_run_dir = Path(tmpdir) / "run_output"
        run = save_run(run, output_dir=local_run_dir)

        s3_run_prefix = f"{eval_prefix}/runs/{run_label}"
        log.info("Uploading results to s3://%s/%s", bucket, s3_run_prefix)
        _upload_directory(local_run_dir, bucket, s3_run_prefix)

        # Also save config snapshot
        config_snapshot = {
            "query_set": args.query_set,
            "top_k": args.top_k,
            "keep_k": args.keep_k,
            "token_budget": args.token_budget,
            "run_generation": args.run_generation,
            "use_llm_judge": args.use_llm_judge,
            "judge_model": args.judge_model if args.use_llm_judge else None,
            "score_ids": args.score_ids,
            "timestamp": timestamp,
        }
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"{s3_run_prefix}/config.json",
            Body=json.dumps(config_snapshot, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REMOTE EVAL RUN")
    print("=" * 72)
    print(f"run_id:        {run.meta.run_id}")
    print(f"query_set:     {args.query_set}")
    print(f"num_queries:   {run.aggregates.overall.num_queries}")
    print(f"mrr:           {run.aggregates.overall.mrr:.4f}")
    print(f"map:           {run.aggregates.overall.map:.4f}")
    for k in sorted(run.aggregates.overall.recall_at_k):
        print(f"recall@{k}:     {run.aggregates.overall.recall_at_k[k]:.4f}")
    print(f"results:       s3://{bucket}/{s3_run_prefix}/")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
```

**Step 2: Verify lint passes**

Run: `./scripts/py -m ruff check scripts/run_remote_eval.py`

**Step 3: Commit**

```
git add scripts/run_remote_eval.py
git commit -m "feat: add run_remote_eval.py for S3-backed eval on ECS"
```

---

### Task 8: Python — `run_remote_query.py` script

Thin wrapper for ad-hoc queries against remote backends.

**Files:**
- Create: `scripts/run_remote_query.py`

**Step 1: Write the script**

Create `scripts/run_remote_query.py`:

```python
"""CLI: Run a single query against remote backends.

Designed to run as an ECS task (query-eval with command override).

Usage:
    ./scripts/py scripts/run_remote_query.py --query "What is 10 CFR 50.46?"
"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from rag import settings
from rag.app.container import build_container
from rag.app.query_runner import run_query

log = logging.getLogger("remote-query")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query remote RAG backends.")
    ap.add_argument(
        "--query",
        default=os.environ.get("QUERY", ""),
        help="Query text (or set QUERY env var).",
    )
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    args = ap.parse_args()

    if not args.query:
        log.error("No query provided. Use --query or set QUERY env var.")
        raise SystemExit(1)

    load_dotenv()
    logging.basicConfig(format="[remote-query] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()
    container = build_container(cfg=cfg)
    container.store.load()

    top_k = args.top_k or cfg.retrieval.top_k
    token_budget = args.token_budget or cfg.context.token_budget

    result = run_query(
        args.query,
        retriever=container.retriever,
        reranker=container.reranker,
        keep_k=cfg.rerank.keep_k,
        context_builder=container.context_builder,
        generator=container.generator,
        logger=container.logger,
        top_k=top_k,
        token_budget=token_budget,
    )

    print(f"\n{result.answer.text}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify lint passes**

Run: `./scripts/py -m ruff check scripts/run_remote_query.py`

**Step 3: Commit**

```
git add scripts/run_remote_query.py
git commit -m "feat: add run_remote_query.py for ad-hoc queries on ECS"
```

---

### Task 9: Shell — `ecs_run_ingest.sh` launcher

Auto-scales workers, launches orchestrator, tails logs, scales workers down.

**Files:**
- Create: `scripts/ecs_run_ingest.sh`

**Step 1: Write the script**

Create `scripts/ecs_run_ingest.sh`:

```bash
#!/usr/bin/env bash
# Launch a distributed ingestion run on ECS.
#
# Usage:
#   scripts/ecs_run_ingest.sh --workers 5 --corpus-id regulations_v1 --index-name regulatory
#
# This script:
#   1. Scales the ingest-worker service to desired count
#   2. Launches the ingest-orchestrator task
#   3. Streams CloudWatch logs
#   4. Scales workers back to 0 when the orchestrator exits

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
WORKERS="${WORKERS:-3}"
CORPUS_ID=""
INDEX_NAME=""
CORPUS_PATH=""
MAX_DOCS=""
EXTRA_ARGS=""

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)      WORKERS="$2"; shift 2 ;;
        --corpus-id)    CORPUS_ID="$2"; shift 2 ;;
        --index-name)   INDEX_NAME="$2"; shift 2 ;;
        --corpus)       CORPUS_PATH="$2"; shift 2 ;;
        --max-docs)     MAX_DOCS="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$CORPUS_ID" || -z "$INDEX_NAME" ]]; then
    echo "Usage: $0 --corpus-id <id> --index-name <name> [--workers N] [--corpus /path]"
    exit 1
fi

WORKER_SERVICE="${CLUSTER}-ingest-worker"
ORCH_TASK_DEF="${CLUSTER}-ingest-orchestrator"

echo "=== Distributed Ingestion ==="
echo "Cluster:      $CLUSTER"
echo "Workers:      $WORKERS"
echo "Corpus ID:    $CORPUS_ID"
echo "Index Name:   $INDEX_NAME"
echo ""

# ── Step 1: Scale up workers ──────────────────────────────────
echo ">>> Scaling ingest-worker service to $WORKERS..."
aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$WORKER_SERVICE" \
    --desired-count "$WORKERS" \
    --no-cli-pager > /dev/null

echo ">>> Waiting for workers to stabilize..."
aws ecs wait services-stable \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE"
echo ">>> $WORKERS workers running."

# ── Step 2: Build command override ────────────────────────────
CMD_ARGS="--corpus-id $CORPUS_ID --index-name $INDEX_NAME"
if [[ -n "$CORPUS_PATH" ]]; then
    CMD_ARGS="$CMD_ARGS --corpus $CORPUS_PATH"
else
    CMD_ARGS="$CMD_ARGS --corpus /data/vault"
fi
if [[ -n "$MAX_DOCS" ]]; then
    CMD_ARGS="$CMD_ARGS --max-docs $MAX_DOCS"
fi

# ── Step 3: Retrieve network config from existing worker ──────
NETWORK_CONFIG=$(aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE" \
    --query 'services[0].networkConfiguration' \
    --output json --no-cli-pager)

SUBNETS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); print(','.join(nc['awsvpcConfiguration']['subnets']))")
SECURITY_GROUPS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); sgs=nc['awsvpcConfiguration'].get('securityGroups',[]); print(','.join(sgs))" 2>/dev/null || echo "")

NETWORK_OVERRIDE="awsvpcConfiguration={subnets=[$SUBNETS],assignPublicIp=ENABLED"
if [[ -n "$SECURITY_GROUPS" ]]; then
    NETWORK_OVERRIDE="$NETWORK_OVERRIDE,securityGroups=[$SECURITY_GROUPS]"
fi
NETWORK_OVERRIDE="$NETWORK_OVERRIDE}"

# ── Step 4: Launch orchestrator task ──────────────────────────
echo ">>> Launching orchestrator task..."
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$ORCH_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "{\"containerOverrides\":[{\"name\":\"ingest-orchestrator\",\"command\":[\"python\",\"scripts/run_orchestrator.py\",$( echo $CMD_ARGS | python3 -c "import sys; args=sys.stdin.read().split(); print(','.join(['\"'+a+'\"' for a in args]))")]}]}" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Orchestrator task: $TASK_ID"

# ── Step 5: Wait for completion ───────────────────────────────
echo ">>> Waiting for orchestrator to complete (tailing logs)..."
LOG_GROUP="/ecs/${CLUSTER}/ingest-orchestrator"

# Wait briefly for the task to start before tailing
sleep 10

# Tail logs in background
aws logs tail "$LOG_GROUP" --follow --format short &
TAIL_PID=$!

# Wait for the task to stop
aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" 2>/dev/null || true

# Stop log tailing
kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

# Check exit code
EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

echo ""
echo ">>> Orchestrator exited with code: $EXIT_CODE"

# ── Step 6: Scale down workers ────────────────────────────────
echo ">>> Scaling ingest-worker service to 0..."
aws ecs update-service \
    --cluster "$CLUSTER" \
    --service "$WORKER_SERVICE" \
    --desired-count 0 \
    --no-cli-pager > /dev/null

echo "=== Done ==="
exit "${EXIT_CODE:-1}"
```

**Step 2: Make executable**

Run: `chmod +x scripts/ecs_run_ingest.sh`

**Step 3: Commit**

```
git add scripts/ecs_run_ingest.sh
git commit -m "feat: add ecs_run_ingest.sh launcher with auto-scaling"
```

---

### Task 10: Shell — `ecs_run_eval.sh` and `ecs_run_query.sh` launchers

**Files:**
- Create: `scripts/ecs_run_eval.sh`
- Create: `scripts/ecs_run_query.sh`

**Step 1: Write `ecs_run_eval.sh`**

Create `scripts/ecs_run_eval.sh`:

```bash
#!/usr/bin/env bash
# Launch an eval run on ECS.
#
# Usage:
#   scripts/ecs_run_eval.sh [--query-set default] [--run-name my-run]

set -euo pipefail

CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
QUERY_SET="${QUERY_SET:-default}"
RUN_NAME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --query-set)  QUERY_SET="$2"; shift 2 ;;
        --run-name)   RUN_NAME="$2"; shift 2 ;;
        --cluster)    CLUSTER="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

EVAL_TASK_DEF="${CLUSTER}-query-eval"
WORKER_SERVICE="${CLUSTER}-ingest-worker"

echo "=== Remote Eval ==="
echo "Cluster:    $CLUSTER"
echo "Query Set:  $QUERY_SET"
echo "Run Name:   ${RUN_NAME:-<auto>}"
echo ""

# Build command
CMD_ARGS="--query-set $QUERY_SET"
if [[ -n "$RUN_NAME" ]]; then
    CMD_ARGS="$CMD_ARGS --run-name $RUN_NAME"
fi

# Retrieve network config from existing service
NETWORK_CONFIG=$(aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE" \
    --query 'services[0].networkConfiguration' \
    --output json --no-cli-pager)

SUBNETS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); print(','.join(nc['awsvpcConfiguration']['subnets']))")
SECURITY_GROUPS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); sgs=nc['awsvpcConfiguration'].get('securityGroups',[]); print(','.join(sgs))" 2>/dev/null || echo "")

NETWORK_OVERRIDE="awsvpcConfiguration={subnets=[$SUBNETS],assignPublicIp=ENABLED"
if [[ -n "$SECURITY_GROUPS" ]]; then
    NETWORK_OVERRIDE="$NETWORK_OVERRIDE,securityGroups=[$SECURITY_GROUPS]"
fi
NETWORK_OVERRIDE="$NETWORK_OVERRIDE}"

# Launch task
echo ">>> Launching eval task..."
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$EVAL_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "{\"containerOverrides\":[{\"name\":\"query-eval\",\"command\":[\"python\",\"scripts/run_remote_eval.py\",$( echo $CMD_ARGS | python3 -c "import sys; args=sys.stdin.read().split(); print(','.join(['\"'+a+'\"' for a in args]))")]}]}" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Eval task: $TASK_ID"

# Tail logs
LOG_GROUP="/ecs/${CLUSTER}/query-eval"
sleep 10
aws logs tail "$LOG_GROUP" --follow --format short &
TAIL_PID=$!

aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" 2>/dev/null || true

kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

echo ">>> Eval exited with code: $EXIT_CODE"
exit "${EXIT_CODE:-1}"
```

**Step 2: Write `ecs_run_query.sh`**

Create `scripts/ecs_run_query.sh`:

```bash
#!/usr/bin/env bash
# Launch an ad-hoc query on ECS.
#
# Usage:
#   scripts/ecs_run_query.sh "What is 10 CFR 50.46?"

set -euo pipefail

CLUSTER="${ECS_CLUSTER:-obsidian-rag}"
QUERY_TEXT="${1:-}"

if [[ -z "$QUERY_TEXT" ]]; then
    echo "Usage: $0 \"your query here\""
    exit 1
fi

EVAL_TASK_DEF="${CLUSTER}-query-eval"
WORKER_SERVICE="${CLUSTER}-ingest-worker"

echo "=== Remote Query ==="
echo "Cluster: $CLUSTER"
echo "Query:   $QUERY_TEXT"
echo ""

# Retrieve network config
NETWORK_CONFIG=$(aws ecs describe-services \
    --cluster "$CLUSTER" \
    --services "$WORKER_SERVICE" \
    --query 'services[0].networkConfiguration' \
    --output json --no-cli-pager)

SUBNETS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); print(','.join(nc['awsvpcConfiguration']['subnets']))")
SECURITY_GROUPS=$(echo "$NETWORK_CONFIG" | python3 -c "import sys,json; nc=json.load(sys.stdin); sgs=nc['awsvpcConfiguration'].get('securityGroups',[]); print(','.join(sgs))" 2>/dev/null || echo "")

NETWORK_OVERRIDE="awsvpcConfiguration={subnets=[$SUBNETS],assignPublicIp=ENABLED"
if [[ -n "$SECURITY_GROUPS" ]]; then
    NETWORK_OVERRIDE="$NETWORK_OVERRIDE,securityGroups=[$SECURITY_GROUPS]"
fi
NETWORK_OVERRIDE="$NETWORK_OVERRIDE}"

# Launch task with command override to run_remote_query.py
echo ">>> Launching query task..."
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$EVAL_TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_OVERRIDE" \
    --overrides "{\"containerOverrides\":[{\"name\":\"query-eval\",\"command\":[\"python\",\"scripts/run_remote_query.py\",\"--query\",\"$QUERY_TEXT\"]}]}" \
    --query 'tasks[0].taskArn' \
    --output text --no-cli-pager)

TASK_ID=$(basename "$TASK_ARN")
echo ">>> Query task: $TASK_ID"

# Tail logs
LOG_GROUP="/ecs/${CLUSTER}/query-eval"
sleep 10
aws logs tail "$LOG_GROUP" --follow --format short &
TAIL_PID=$!

aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" 2>/dev/null || true

kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text --no-cli-pager)

echo ">>> Query exited with code: $EXIT_CODE"
exit "${EXIT_CODE:-1}"
```

**Step 3: Make executable**

Run: `chmod +x scripts/ecs_run_eval.sh scripts/ecs_run_query.sh`

**Step 4: Commit**

```
git add scripts/ecs_run_eval.sh scripts/ecs_run_query.sh
git commit -m "feat: add ecs_run_eval.sh and ecs_run_query.sh launchers"
```

---

### Task 11: Makefile — Add remote operation targets

**Files:**
- Modify: `Makefile` (append after `ecs-status` target, line ~214)

**Step 1: Add targets**

Add to `Makefile` after the `ecs-status` target:

```makefile
# -------------------------------------------------------------------
# Remote Operations (ECS)
# -------------------------------------------------------------------

WORKERS ?= 3
CORPUS_ID ?= regulations_v1
INDEX_NAME ?= regulatory
QUERY_SET ?= default
RUN_NAME ?=

ingest-remote:  ## Run distributed ingestion on ECS (auto-scales workers)
	scripts/ecs_run_ingest.sh \
		--workers $(WORKERS) \
		--corpus-id $(CORPUS_ID) \
		--index-name $(INDEX_NAME)

eval-remote:  ## Run eval against remote backends on ECS
	scripts/ecs_run_eval.sh \
		--query-set $(QUERY_SET) \
		$(if $(RUN_NAME),--run-name $(RUN_NAME),)

query-remote:  ## Run ad-hoc query on ECS
	scripts/ecs_run_query.sh "$(QUERY)"

upload-eval-queries:  ## Sync local eval datasets to S3
	@BUCKET=$$(cd infra && terraform output -raw corpus_bucket_name 2>/dev/null || echo "obsidian-rag-corpus"); \
	echo "Uploading eval datasets to s3://$$BUCKET/eval/queries/default/"; \
	aws s3 sync eval/datasets/ "s3://$$BUCKET/eval/queries/default/" --exclude "*.pyc"
```

Also add the new targets to the `.PHONY` declaration on line 1:

Update line 1-6 to include:
```makefile
.PHONY: help index index-dummy ask ask-dummy results verdict tail-logs clean-index \
        index-regulatory index-regulatory-dummy normalize-regulatory \
        test lint fmt typecheck env-check \
        docker-build docker-up docker-down \
        infra-init infra-plan infra-apply infra-destroy \
        ecs-up ecs-down ecs-status \
        ingest-remote eval-remote query-remote upload-eval-queries
```

**Step 2: Commit**

```
git add Makefile
git commit -m "feat: add remote operation Make targets"
```

---

### Task 12: Run tests + lint

Final validation that everything compiles, lints, and existing tests still pass.

**Files:** None (validation only)

**Step 1: Run full test suite**

Run: `./scripts/py -m pytest -q`
Expected: All tests pass (including new orchestrator tests)

**Step 2: Run linter**

Run: `./scripts/py -m ruff check .`
Expected: No errors

**Step 3: Run type checker**

Run: `./scripts/py -m mypy --config-file pyproject.toml src`
Expected: No new errors

**Step 4: Validate Terraform**

Run: `cd infra && terraform validate`
Expected: "Success! The configuration is valid."
