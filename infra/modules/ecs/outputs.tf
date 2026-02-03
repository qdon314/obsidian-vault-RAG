output "cluster_id" {
  description = "ECS cluster ID"
  value       = aws_ecs_cluster.this.id
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.this.name
}

output "app_service_name" {
  description = "Name of the RAG app ECS service"
  value       = aws_ecs_service.app.name
}

output "qdrant_service_name" {
  description = "Name of the Qdrant ECS service"
  value       = aws_ecs_service.qdrant.name
}

output "qdrant_discovery_name" {
  description = "DNS name for Qdrant via Cloud Map"
  value       = "qdrant.${var.cluster_name}.local"
}
