output "ecr_repository_url" {
  description = "ECR repository URL (for docker push)"
  value       = module.ecr.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket for artifacts"
  value       = module.s3.bucket_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "qdrant_dns" {
  description = "Qdrant service discovery DNS name"
  value       = module.ecs.qdrant_discovery_name
}

output "sqs_queue_url" {
  description = "SQS queue URL for ingestion tasks"
  value       = module.sqs.queue_url
}

output "sqs_queue_arn" {
  description = "SQS queue ARN for ingestion tasks"
  value       = module.sqs.queue_arn
}

output "rds_endpoint" {
  description = "RDS endpoint (host:port)"
  value       = module.rds.endpoint
}

output "rds_dsn_parameter_arn" {
  description = "SSM parameter ARN containing RDS DSN"
  value       = aws_ssm_parameter.rds_dsn.arn
}

output "corpus_bucket_name" {
  description = "S3 bucket for distributed ingestion corpus-of-record"
  value       = local.corpus_bucket_name
}

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
