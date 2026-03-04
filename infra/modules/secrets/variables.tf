variable "openai_api_key" {
  description = "OpenAI API key (stored as SecureString)"
  type        = string
  sensitive   = true
}

variable "scaledown_api_key" {
  description = "ScaleDown API key for prompt compression (stored as SecureString)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "name_prefix" {
  description = "Prefix for parameter names"
  type        = string
  default     = "/rag"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
