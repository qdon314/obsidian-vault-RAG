output "queue_url" {
  value = aws_sqs_queue.ingest_tasks.url
}

output "queue_arn" {
  value = aws_sqs_queue.ingest_tasks.arn
}

output "dlq_url" {
  value = aws_sqs_queue.ingest_dlq.url
}
