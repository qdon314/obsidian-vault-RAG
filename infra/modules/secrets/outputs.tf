output "openai_api_key_arn" {
  description = "ARN of the OpenAI API key parameter"
  value       = aws_ssm_parameter.openai_api_key.arn
}

output "openai_api_key_name" {
  description = "Name of the OpenAI API key parameter"
  value       = aws_ssm_parameter.openai_api_key.name
}
