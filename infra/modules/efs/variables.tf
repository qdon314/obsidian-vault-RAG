variable "name_prefix" {
  description = "Prefix for EFS resource names"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for EFS mount targets (one per AZ)"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs allowed to access EFS"
  type        = list(string)
}

variable "vpc_id" {
  description = "VPC ID for the EFS security group"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
