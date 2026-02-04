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
