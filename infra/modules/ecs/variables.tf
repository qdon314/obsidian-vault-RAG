variable "cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
}

variable "app_image" {
  description = "Docker image URI for the RAG app (from ECR)"
  type        = string
}

variable "qdrant_image" {
  description = "Docker image for Qdrant"
  type        = string
  default     = "qdrant/qdrant:v1.13.2"
}

variable "openai_api_key_arn" {
  description = "ARN of the SSM parameter containing the OpenAI API key"
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the S3 bucket for artifact storage"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for ECS tasks"
  type        = list(string)
  default     = []
}

variable "app_cpu" {
  description = "CPU units for RAG app task (1024 = 1 vCPU)"
  type        = number
  default     = 256
}

variable "app_memory" {
  description = "Memory (MB) for RAG app task"
  type        = number
  default     = 512
}

variable "qdrant_cpu" {
  description = "CPU units for Qdrant task"
  type        = number
  default     = 512
}

variable "qdrant_memory" {
  description = "Memory (MB) for Qdrant task"
  type        = number
  default     = 1024
}

variable "app_desired_count" {
  description = "Desired number of RAG app tasks (set to 0 for scale-to-zero)"
  type        = number
  default     = 0
}

variable "qdrant_desired_count" {
  description = "Desired number of Qdrant tasks (set to 0 for scale-to-zero)"
  type        = number
  default     = 0
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
