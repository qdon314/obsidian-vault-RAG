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

variable "qdrant_collection" {
  description = "Qdrant collection name used by all ECS tasks"
  type        = string
  default     = "obsidian"
}

variable "openai_api_key_arn" {
  description = "ARN of the SSM parameter containing the OpenAI API key"
  type        = string
}

variable "scaledown_api_key_arn" {
  description = "ARN of the SSM parameter containing the ScaleDown API key (empty = compression disabled)"
  type        = string
  default     = ""
}

variable "s3_bucket_arn" {
  description = "ARN of the S3 bucket for artifact storage"
  type        = string
}

variable "corpus_bucket_arn" {
  description = "ARN of the S3 bucket used as corpus source of truth"
  type        = string
  default     = ""
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

# Worker-specific variables
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

variable "worker_desired_count" {
  description = "Desired number of ingest worker tasks"
  type        = number
  default     = 0
}

variable "sqs_queue_url" {
  description = "URL of the SQS queue for ingestion tasks"
  type        = string
  default     = ""
}

variable "sqs_queue_arn" {
  description = "ARN of the SQS queue for ingestion tasks"
  type        = string
  default     = ""
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for corpus storage"
  type        = string
  default     = ""
}

variable "rds_dsn_arn" {
  description = "ARN of the SSM parameter containing the RDS DSN"
  type        = string
  default     = ""
}

variable "corpus_s3_prefix" {
  description = "Prefix for raw corpus objects in S3"
  type        = string
  default     = "corpus"
}

variable "chunk_s3_prefix" {
  description = "Prefix for chunk-store objects in S3"
  type        = string
  default     = "chunks"
}

variable "chunk_max_s3_workers" {
  description = "Max S3 workers for chunk storage/hydration"
  type        = number
  default     = 4
}

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

variable "qdrant_efs_file_system_id" {
  description = "EFS file system ID for Qdrant persistent storage (empty = no EFS)"
  type        = string
  default     = ""
}

variable "qdrant_efs_access_point_id" {
  description = "EFS access point ID for Qdrant storage"
  type        = string
  default     = ""
}
