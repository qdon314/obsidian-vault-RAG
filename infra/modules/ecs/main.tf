resource "aws_ecs_cluster" "this" {
  name = var.cluster_name
  tags = var.tags
}

# --- CloudWatch log groups with retention ---
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.cluster_name}/app"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "qdrant" {
  name              = "/ecs/${var.cluster_name}/qdrant"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# --- Cloud Map namespace for service discovery ---
resource "aws_service_discovery_private_dns_namespace" "this" {
  name = "${var.cluster_name}.local"
  vpc  = data.aws_subnet.first.vpc_id
  tags = var.tags
}

data "aws_subnet" "first" {
  id = var.subnet_ids[0]
}

# --- Qdrant service discovery ---
resource "aws_service_discovery_service" "qdrant" {
  name = "qdrant"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.this.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# --- Qdrant task definition ---
resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${var.cluster_name}-qdrant"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.qdrant_cpu
  memory                   = var.qdrant_memory
  execution_role_arn       = aws_iam_role.task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "qdrant"
      image     = var.qdrant_image
      essential = true
      portMappings = [
        { containerPort = 6333, protocol = "tcp" },
        { containerPort = 6334, protocol = "tcp" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.qdrant.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "qdrant"
        }
      }
    }
  ])

  tags = var.tags
}

data "aws_region" "current" {}

# --- Qdrant service ---
resource "aws_ecs_service" "qdrant" {
  name            = "${var.cluster_name}-qdrant"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.qdrant.arn
  desired_count   = var.qdrant_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true
  }

  service_registries {
    registry_arn = aws_service_discovery_service.qdrant.arn
  }

  tags = var.tags
}

# --- RAG app task definition ---
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.cluster_name}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "rag-app"
      image     = var.app_image
      essential = true
      environment = [
        { name = "RAG_VECTORSTORE__BACKEND", value = "qdrant" },
        { name = "RAG_VECTORSTORE__QDRANT_URL", value = "http://qdrant.${var.cluster_name}.local:6333" },
        { name = "RAG_VECTORSTORE__QDRANT_COLLECTION", value = "obsidian" },
      ]
      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = var.openai_api_key_arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "app"
        }
      }
    }
  ])

  tags = var.tags
}

# --- RAG app service ---
resource "aws_ecs_service" "app" {
  name            = "${var.cluster_name}-app"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.app_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true
  }

  tags = var.tags
}

# --- Ingest worker task definition ---
resource "aws_ecs_task_definition" "ingest_worker" {
  family                   = "${var.cluster_name}-ingest-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "ingest-worker"
      image     = var.app_image
      essential = true
      command   = ["python", "scripts/run_worker.py"]
      environment = [
        { name = "RAG_VECTORSTORE__BACKEND", value = "qdrant" },
        { name = "RAG_VECTORSTORE__QDRANT_URL", value = "http://qdrant.${var.cluster_name}.local:6333" },
        { name = "RAG_DISTRIBUTED_INGESTION__ENABLED", value = "true" },
        { name = "RAG_DISTRIBUTED_INGESTION__SQS_QUEUE_URL", value = var.sqs_queue_url },
        { name = "RAG_DISTRIBUTED_INGESTION__CORPUS_S3_BUCKET", value = var.s3_bucket_name },
        { name = "RAG_DISTRIBUTED_INGESTION__CORPUS_S3_PREFIX", value = var.corpus_s3_prefix },
        { name = "RAG_CHUNK_STORAGE__BACKEND", value = "s3" },
        { name = "RAG_CHUNK_STORAGE__S3_BUCKET", value = var.s3_bucket_name },
        { name = "RAG_CHUNK_STORAGE__S3_PREFIX", value = var.chunk_s3_prefix },
        { name = "RAG_CHUNK_STORAGE__MAX_S3_WORKERS", value = tostring(var.chunk_max_s3_workers) },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = var.openai_api_key_arn },
        { name = "RAG_DISTRIBUTED_INGESTION__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
        { name = "RAG_CHUNK_STORAGE__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.worker.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "worker"
        }
      }
    }
  ])

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${var.cluster_name}/ingest-worker"
  retention_in_days = 30
  tags              = var.tags
}

# --- Ingest worker service ---
resource "aws_ecs_service" "ingest_worker" {
  name            = "${var.cluster_name}-ingest-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.ingest_worker.arn
  desired_count   = var.worker_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true
  }

  tags = var.tags
}
