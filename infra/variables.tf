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
