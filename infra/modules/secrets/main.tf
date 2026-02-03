# Note: The secret value is stored in Terraform state in plaintext.
# For production, consider managing the value out of band (aws ssm put-parameter)
# and using lifecycle { ignore_changes = [value] } to keep it out of state.
resource "aws_ssm_parameter" "openai_api_key" {
  name        = "${var.name_prefix}/openai-api-key"
  description = "OpenAI API key for the RAG pipeline"
  type        = "SecureString"
  value       = var.openai_api_key

  tags = var.tags
}
