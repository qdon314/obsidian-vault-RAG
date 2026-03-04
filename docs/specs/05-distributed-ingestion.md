# Distributed Ingestion Specification

## Overview

The distributed ingestion system enables scalable, parallel document processing for the RAG pipeline. It separates the control plane (enumerator) from the worker plane, using AWS services for storage, messaging, and persistence.

## Goals

1. **Scalability**: Process thousands of documents in parallel
2. **Reliability**: Lease-based work acquisition prevents duplicate processing
3. **Observability**: Full job/task tracking in Postgres
4. **Cost Efficiency**: Scale workers to zero when idle

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   CLI Tool   │───▶│  Enumerator  │───▶│  Job Store   │       │
│  │start_ingest  │    │   Service    │    │  (Postgres)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AWS INFRASTRUCTURE                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  S3 Bucket   │    │  SQS Queue   │    │  RDS Postgres│       │
│  │Corpus Store  │    │Task Queue    │    │Job/Task State│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WORKER FLEET                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Worker 1   │    │   Worker 2   │    │   Worker N   │       │
│  │run_worker.py │    │run_worker.py │    │run_worker.py │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Job Creation** (Enumerator):
   - CLI invokes `start_ingestion.py` with S3 corpus prefix
   - Enumerator creates `IngestJob` record in Postgres
   - Lists all objects in S3 prefix
   - Creates `DocumentRecord` and `IngestTask` records
   - Enqueues task messages to SQS

2. **Task Processing** (Workers):
   - Workers poll SQS for messages
   - On message receipt, lease task from Postgres (atomic update)
   - Fetch document content from S3
   - Process: chunk → embed → upsert to VectorStore
   - Mark task complete in Postgres
   - Delete message from SQS

3. **Lease Management**:
   - Tasks have `leased_at` and `leased_by` fields
   - Lease expires after configurable timeout (default: 5 minutes)
   - Expired leases can be re-acquired by other workers

## Domain Models

### IngestJob

```python
@dataclass(frozen=True)
class IngestJob:
    job_id: str           # UUID
    corpus_prefix: str    # S3 URI (s3://bucket/prefix/)
    status: JobStatus     # PENDING, RUNNING, COMPLETED, FAILED
    created_at: datetime
    updated_at: datetime
    metadata: dict        # Optional tracing info
```

### IngestTask

```python
@dataclass(frozen=True)
class IngestTask:
    task_id: str          # UUID
    job_id: str           # Parent job reference
    s3_key: str           # S3 object key
    status: TaskStatus    # PENDING, LEASED, COMPLETED, FAILED
    leased_at: datetime | None
    leased_by: str | None # Worker identifier
    completed_at: datetime | None
    error_message: str | None
```

### DocumentRecord

```python
@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str           # Content hash
    job_id: str           # Parent job reference
    s3_key: str           # S3 object key
    etag: str             # S3 ETag for versioning
    size_bytes: int       # Object size
```

## Ports

### IngestJobStore

```python
@runtime_checkable
class IngestJobStore(Protocol):
    """Port for job/task persistence."""

    def create_job(self, corpus_prefix: str, metadata: dict | None = None) -> IngestJob: ...
    def get_job(self, job_id: str) -> IngestJob | None: ...
    def update_job_status(self, job_id: str, status: JobStatus) -> None: ...

    def add_document_records(self, job_id: str, records: list[DocumentRecord]) -> None: ...
    def get_document_record(self, doc_id: str) -> DocumentRecord | None: ...

    def add_tasks(self, job_id: str, tasks: list[IngestTask]) -> None: ...
    def lease_task(self, worker_id: str, lease_timeout: int = 300) -> IngestTask | None: ...
    def complete_task(self, task_id: str, worker_id: str) -> bool: ...
    def fail_task(self, task_id: str, error: str) -> bool: ...
    def get_task_stats(self, job_id: str) -> TaskStats: ...
```

### RawDocumentStore

```python
@runtime_checkable
class RawDocumentStore(Protocol):
    """Port for raw document storage (S3)."""

    def list_objects(self, prefix: str) -> Iterator[ObjectInfo]: ...
    def get_object(self, key: str) -> bytes: ...
    def put_object(self, key: str, data: bytes) -> None: ...
```

### TaskQueue

```python
@runtime_checkable
class TaskQueue(Protocol):
    """Port for task distribution (SQS)."""

    def send_message(self, body: str, *, delay_seconds: int = 0) -> str: ...
    def receive_messages(self, max_messages: int = 10, wait_time_seconds: int = 20) -> list[QueueMessage]: ...
    def delete_message(self, receipt_handle: str) -> None: ...
    def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None: ...
```

## Adapters

### PostgresIngestJobStore

- Implements `IngestJobStore` using PostgreSQL
- Uses atomic `UPDATE ... WHERE` for lease acquisition
- Connection pooling via `psycopg2`

### S3RawDocumentStore

- Implements `RawDocumentStore` using boto3
- Supports pagination for large object lists
- Streaming GET for memory efficiency

### SqsTaskQueue

- Implements `TaskQueue` using AWS SQS
- Long-polling with `WaitTimeSeconds`
- Visibility timeout management for lease safety

## Configuration

```toml
[distributed_ingestion]
enabled = false
s3_bucket = ""
s3_prefix = ""
sqs_queue_url = ""
postgres_dsn = ""
lease_timeout_seconds = 300
max_s3_workers = 4
```

## CLI

### start_ingestion.py

```bash
./scripts/py scripts/start_ingestion.py \
    --corpus-prefix s3://my-bucket/corpus/ \
    [--settings settings.toml]
```

Creates a new ingestion job and returns the `job_id`.

### run_worker.py

```bash
./scripts/py scripts/run_worker.py \
    [--max-idle 300] \
    [--settings settings.toml]
```

Polls SQS and processes tasks until idle timeout reached.

## Database Schema

```sql
-- Jobs table
CREATE TABLE ingest_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_prefix TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Document records table
CREATE TABLE document_records (
    doc_id TEXT PRIMARY KEY,
    job_id UUID REFERENCES ingest_jobs(job_id),
    s3_key TEXT NOT NULL,
    etag TEXT NOT NULL,
    size_bytes BIGINT NOT NULL
);

-- Tasks table
CREATE TABLE ingest_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES ingest_jobs(job_id),
    s3_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    leased_at TIMESTAMPTZ,
    leased_by TEXT,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_tasks_job_status ON ingest_tasks(job_id, status);
CREATE INDEX idx_tasks_leased ON ingest_tasks(leased_at) WHERE status = 'LEASED';
```

## Error Handling

| Error Type | Behavior |
|------------|----------|
| S3 fetch failure | Task marked FAILED, message returned to queue |
| Embedding failure | Task marked FAILED, message returned to queue |
| Worker crash | Lease expires, task re-acquired by another worker |
| Lease timeout | Task becomes available for re-lease |

## Monitoring

Track ingestion progress via SQL:

```sql
-- Job status overview
SELECT 
    j.job_id,
    j.status,
    COUNT(t.task_id) FILTER (WHERE t.status = 'PENDING') as pending,
    COUNT(t.task_id) FILTER (WHERE t.status = 'LEASED') as leased,
    COUNT(t.task_id) FILTER (WHERE t.status = 'COMPLETED') as completed,
    COUNT(t.task_id) FILTER (WHERE t.status = 'FAILED') as failed
FROM ingest_jobs j
LEFT JOIN ingest_tasks t ON j.job_id = t.job_id
WHERE j.job_id = '...'
GROUP BY j.job_id, j.status;
```

## Future Enhancements

1. **Dead Letter Queue**: Failed tasks move to DLQ after N retries
2. **Progress Callbacks**: Webhook notifications on job completion
3. **Incremental Ingestion**: ETag-based change detection
4. **Priority Queues**: High-priority documents processed first
