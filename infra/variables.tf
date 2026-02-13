variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name (used as prefix for resource names)"
  type        = string
  default     = "obsidian-rag"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
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

variable "app_desired_count" {
  description = "Desired RAG app task count (0 = scaled down)"
  type        = number
  default     = 0
}

variable "qdrant_desired_count" {
  description = "Desired Qdrant task count (0 = scaled down)"
  type        = number
  default     = 0
}

variable "worker_desired_count" {
  description = "Desired ingest worker task count (0 = scaled down)"
  type        = number
  default     = 0
}

variable "worker_cpu" {
  description = "CPU units for ingest worker task"
  type        = number
  default     = 256
}

variable "worker_memory" {
  description = "Memory (MB) for ingest worker task"
  type        = number
  default     = 512
}

variable "db_username" {
  description = "RDS Postgres username"
  type        = string
  default     = "rag"
}

variable "db_password" {
  description = "RDS Postgres password"
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t4g.micro"
}

variable "sqs_visibility_timeout" {
  description = "SQS visibility timeout in seconds"
  type        = number
  default     = 300
}

variable "sqs_max_receive_count" {
  description = "SQS max receive count before DLQ"
  type        = number
  default     = 5
}

variable "corpus_bucket_name" {
  description = "Optional S3 bucket name for raw corpus-of-record; defaults to artifacts bucket"
  type        = string
  default     = ""
}

variable "corpus_s3_prefix" {
  description = "Prefix for raw corpus objects in corpus bucket"
  type        = string
  default     = "corpus"
}

variable "chunk_s3_prefix" {
  description = "Prefix for chunk-store objects in chunk storage bucket"
  type        = string
  default     = "chunks"
}

variable "chunk_max_s3_workers" {
  description = "Max S3 workers for chunk hydration/storage in worker tasks"
  type        = number
  default     = 4
}
