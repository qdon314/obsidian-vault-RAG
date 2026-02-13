# Infrastructure Modules

This directory contains reusable Terraform modules for the AWS deployment.

## Modules

### `modules/rds`

Purpose:

- Provision Postgres (RDS) for distributed ingestion job/task persistence.

Inputs:

- `name_prefix`
- `subnet_ids`
- `security_group_ids`
- `instance_class` (default `db.t4g.micro`)
- `db_username` (default `rag`)
- `db_password` (sensitive)
- `tags`

Outputs:

- `endpoint` (host:port)
- `dsn` (sensitive connection string)

Notes:

- Used by distributed ingestion (`IngestJobStore`) not by online query path.
- Keep DB private (`publicly_accessible = false`) and restrict SG ingress to worker/task networks.

### `modules/sqs`

Purpose:

- Provision task queue and dead-letter queue for distributed ingestion workers.

Inputs:

- `name_prefix`
- `visibility_timeout` (default `300`)
- `max_receive_count` (default `5`)
- `tags`

Outputs:

- `queue_url`
- `queue_arn`
- `dlq_url`

Notes:

- Visibility timeout should align with worker lease settings.
- DLQ receives messages after repeated delivery failures.

### `modules/ecs`

Purpose:

- Provision app, qdrant, and ingestion worker ECS resources.

Distributed-ingestion-related inputs:

- `worker_cpu`
- `worker_memory`
- `worker_desired_count`
- `sqs_queue_url`
- `sqs_queue_arn`
- `s3_bucket_name`
- `rds_dsn_arn`

Notes:

- Worker task uses `scripts/run_worker.py`.
- IAM includes SQS permissions scoped to `sqs_queue_arn`.
- `rds_dsn_arn` should reference a secure parameter/secret containing Postgres DSN.

## Wiring Summary

At root level (`infra/main.tf`):

1. Create SQS and RDS modules.
2. Pass module outputs (`queue_url`, `queue_arn`, `dsn`/secret ARN) into ECS module.
3. Set worker desired count to scale worker fleet.

## Security Assumptions

- Secrets are injected from secure parameter stores, not plaintext in task definitions.
- RDS is private and reachable only from trusted network paths.
- S3 access is least-privilege scoped to required bucket/prefix.
