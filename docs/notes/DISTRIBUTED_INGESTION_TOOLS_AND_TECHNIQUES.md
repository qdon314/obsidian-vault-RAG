# Distributed Ingestion: Tools and Techniques

This note explains the concrete tools and engineering techniques used in the Phase 3 distributed ingestion changes.

## 1. Core Tools

### Python and adapters

- `boto3` for AWS S3 and SQS clients
- `psycopg2` connection pooling for Postgres job/task persistence
- Dataclass-based domain models (`IngestJob`, `IngestTask`, `DocumentRecord`)
- Protocol ports for boundaries:
  - `IngestJobStore`
  - `RawDocumentStore`
  - `TaskQueue`

### Runtime entry points

- `scripts/start_ingestion.py` (enumerator/control plane)
- `scripts/run_worker.py` (worker/data plane)
- `src/rag/app/ingestion/enumerator.py`
- `src/rag/app/ingestion/worker.py`
- `src/rag/app/ingestion/worker_loop.py`

### Infrastructure

- Terraform modules:
  - `infra/modules/sqs`
  - `infra/modules/rds`
  - `infra/modules/ecs` (worker task/service updates)
- ECS/Fargate for worker execution
- CloudWatch logs for app/qdrant/worker visibility

## 2. Architecture Technique: Ports and Adapters

The changes follow the existing hexagonal architecture:

- Domain layer defines ingestion state models
- Ports define behavior contracts
- Adapters implement external systems (S3/SQS/Postgres)
- Application layer orchestrates workflows (enumerator + worker loop)

Benefits:

- Testability with fakes/mocks
- Swappable infrastructure adapters
- Clear separation between orchestration and IO concerns

## 3. Reliability Techniques

### Lease-based work claiming

Workers claim tasks with a lease window (`lease_expires_at`) to prevent duplicate active work.

- Claimable: `PENDING`, `RETRYABLE`
- Reclaimable: `RUNNING` with expired lease

### Message-to-task binding

Task acquisition is constrained by `job_id + doc_id` to avoid cross-document mismatch when queue delivery is out of order or retried.

### Explicit ack/nack semantics

- `ack`: processing succeeded or work already completed
- `nack`: processing failed or required state missing (for retry path)

### Idempotent persistence

- `upsert_document()` for corpus records
- `create_tasks()` with conflict protection on `(job_id, doc_id)`

## 4. Data Consistency Techniques

### Corpus-of-record in S3

Enumerator stores raw documents in S3 before worker processing, so workers process stable source data independent of local disk.

### Deterministic object keying

Raw document keys use stable hash-derived paths for repeatability and distribution.

### Dual write in processing path

Worker writes both:

- Vector store (retrieval)
- Chunk store (hydration/source payload)

## 5. Operational Techniques

### Config-driven behavior

Distributed ingestion is gated behind `[distributed_ingestion]` settings.

Required runtime settings:

- `enabled`
- `postgres_dsn`
- `sqs_queue_url`
- `corpus_s3_bucket`

### Environment override pattern

Deployment uses `RAG_<SECTION>__<KEY>` environment variables to avoid hardcoding environment-specific values in committed config.

### Scale-to-zero pattern

Worker desired count can remain `0` when idle and scale up only during ingestion runs.

## 6. Testing Techniques

### Contract-style tests with fakes

- `FakeIngestJobStore`
- `FakeRawDocumentStore`
- `FakeTaskQueue`

These validate orchestration behavior without AWS or Postgres dependencies.

### Adapter unit tests with mocks

- Mocked SQS client tests for `SQSTaskQueue`
- Mocked pool/cursor tests for `PostgresIngestJobStore`
- Mocked S3 client tests for `S3RawDocumentStore`

### Integration smoke test

`tests/integration/test_distributed_ingestion.py` exercises enumerator -> queue -> worker flow with fakes to validate end-to-end orchestration.

## 7. Key Design Tradeoffs

- Postgres + SQS chosen for straightforward debugging and operational familiarity
- At-least-once delivery accepted, with idempotency + lease rules to keep outcomes correct
- Lightweight control-plane model favored over adding a heavy workflow engine
