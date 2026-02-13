# Distributed Ingestion Runbook

Operational guide for running and troubleshooting distributed ingestion.

## Scope

This runbook covers:

- Enumerator startup (`scripts/start_ingestion.py`)
- Worker startup (`scripts/run_worker.py`)
- SQS/Postgres/S3 interactions
- Common failure modes and recovery actions

## Required Configuration

Set these values in `settings.toml` (or via env overrides):

```toml
[distributed_ingestion]
enabled = true
postgres_dsn = "postgresql://user:pass@host:5432/rag"
sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
corpus_s3_bucket = "rag-prod-artifacts"
corpus_s3_prefix = "corpus"
worker_lease_duration_s = 300
max_task_retries = 3
```

Required for execution:

- `OPENAI_API_KEY` (if using OpenAI embeddings/LLM)
- Network access to Postgres, SQS, S3, and vector store backend

## Start a Job

```bash
./scripts/py scripts/start_ingestion.py \
  --corpus /path/to/corpus \
  --corpus-id my-corpus \
  --index-name my-index
```

What this does:

1. Creates an `IngestJob` record
2. Stores raw documents in S3
3. Upserts `DocumentRecord` rows
4. Creates `IngestTask` rows (`PENDING`)
5. Enqueues SQS task messages
6. Marks job `RUNNING`

## Start Workers

```bash
./scripts/py scripts/run_worker.py --worker-id worker-1
```

Run multiple workers with unique `--worker-id` values.

Worker loop behavior:

- Receives one SQS message (`job_id`, `corpus_id`, `doc_id`)
- Leases task by `job_id + doc_id`
- Processes raw doc (chunk -> embed -> upsert)
- Marks task `SUCCEEDED` or `RETRYABLE`
- `ack` on success, `nack` on failure

## Task/Lease Semantics

- Claimable tasks:
  - `PENDING`
  - `RETRYABLE`
- Reclaimable tasks:
  - `RUNNING` with expired `lease_expires_at`
- Lease ownership prevents duplicate processing
- Message/doc mismatch is treated as failure and nacked

## Operational Checks

Use these checks during active jobs:

1. Queue depth is decreasing
2. `ingest_tasks` shows increasing `SUCCEEDED`
3. `RETRYABLE` is not growing unbounded
4. Worker logs show steady completions, not repeated failures

## Failure Modes and Recovery

### No workers processing

Symptoms:

- SQS depth remains high
- Tasks remain `PENDING`

Actions:

1. Verify worker service desired count > 0
2. Confirm workers can reach Postgres/SQS/S3
3. Confirm `distributed_ingestion.enabled = true`

### Tasks stuck in `RUNNING`

Symptoms:

- Many `RUNNING` tasks, little progress

Actions:

1. Confirm worker crashes/restarts in logs
2. Verify lease duration is reasonable (`worker_lease_duration_s`)
3. Wait for lease expiry; tasks become reclaimable automatically

### High `RETRYABLE` count

Symptoms:

- Repeated retries with same error

Actions:

1. Inspect `last_error` in task records
2. Fix root cause (missing raw doc, model/provider outage, vector store outage)
3. Restart workers after fix

### Missing document record

Symptoms:

- Worker logs: missing record for `corpus_id/doc_id`
- Frequent nacks

Actions:

1. Verify enumerator completed document upserts
2. Verify message body fields (`job_id`, `corpus_id`, `doc_id`) are valid
3. Re-enqueue only after record integrity is restored

## Safe Rollback / Disable

To stop new distributed jobs:

1. Set `[distributed_ingestion].enabled = false`
2. Stop worker services
3. Leave existing data in S3/Postgres for postmortem/restart

## Related Docs

- `docs/ARCHITECTURE.md` (Distributed Ingestion section)
- `docs/CONFIGURATION.md` (`[distributed_ingestion]`)
- `docs/specs/05-distributed-ingestion.md`
