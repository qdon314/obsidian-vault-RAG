variable "openai_api_key" {
  description = "OpenAI API key (stored as SecureString)"
  type        = string
  sensitive   = true
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
