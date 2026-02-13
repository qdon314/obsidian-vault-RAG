resource "aws_sqs_queue" "ingest_tasks" {
  name                       = "${var.name_prefix}-ingest-tasks"
  visibility_timeout_seconds = var.visibility_timeout
  message_retention_seconds  = 1209600 # 14 days
  receive_wait_time_seconds  = 20      # long polling
  tags                       = var.tags
}

resource "aws_sqs_queue" "ingest_dlq" {
  name = "${var.name_prefix}-ingest-tasks-dlq"
  tags = var.tags
}

resource "aws_sqs_queue_redrive_policy" "this" {
  queue_url = aws_sqs_queue.ingest_tasks.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingest_dlq.arn
    maxReceiveCount     = var.max_receive_count
  })
}
