output "openai_api_key_arn" {
  description = "ARN of the OpenAI API key parameter"
  value       = aws_ssm_parameter.openai_api_key.arn
}

output "openai_api_key_name" {
  description = "Name of the OpenAI API key parameter"
  value       = aws_ssm_parameter.openai_api_key.name
}

output "scaledown_api_key_arn" {
  description = "ARN of the ScaleDown API key parameter (empty string if not configured)"
  value       = length(aws_ssm_parameter.scaledown_api_key) > 0 ? aws_ssm_parameter.scaledown_api_key[0].arn : ""
}
