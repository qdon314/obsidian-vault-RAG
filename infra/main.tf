terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # For production, configure remote state:
  # backend "s3" {
  #   bucket = "obsidian-rag-tfstate"
  #   key    = "infra/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

locals {
  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
  corpus_bucket_name        = var.corpus_bucket_name != "" ? var.corpus_bucket_name : module.s3.bucket_name
  legacy_security_group_ids = var.security_group_ids != null ? var.security_group_ids : []
  ecs_security_group_ids    = length(var.ecs_security_group_ids) > 0 ? var.ecs_security_group_ids : local.legacy_security_group_ids
  rds_security_group_ids    = length(var.rds_security_group_ids) > 0 ? var.rds_security_group_ids : local.legacy_security_group_ids
}

data "aws_subnet" "first" {
  id = var.subnet_ids[0]
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-sg"
  description = "Security group for ECS tasks"
  vpc_id      = data.aws_subnet.first.vpc_id
  tags        = local.tags
}

resource "aws_security_group_rule" "ecs_egress_all" {
  type              = "egress"
  description       = "Allow ECS tasks outbound access to AWS APIs and internet"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.ecs_tasks.id
}

resource "aws_security_group_rule" "ecs_ingress_qdrant_http_from_self" {
  type                     = "ingress"
  description              = "Allow ECS tasks to reach Qdrant HTTP"
  from_port                = 6333
  to_port                  = 6333
  protocol                 = "tcp"
  security_group_id        = aws_security_group.ecs_tasks.id
  source_security_group_id = aws_security_group.ecs_tasks.id
}

resource "aws_security_group_rule" "ecs_ingress_qdrant_grpc_from_self" {
  type                     = "ingress"
  description              = "Allow ECS tasks to reach Qdrant gRPC"
  from_port                = 6334
  to_port                  = 6334
  protocol                 = "tcp"
  security_group_id        = aws_security_group.ecs_tasks.id
  source_security_group_id = aws_security_group.ecs_tasks.id
}

resource "aws_security_group" "rds" {
  name = "${var.project_name}-rds-sg"
  # Keep description aligned with existing imported SG to avoid ForceNew replacement.
  description = "RDS access for ingestion bootstrap"
  vpc_id      = data.aws_subnet.first.vpc_id
  tags        = local.tags
}

resource "aws_security_group_rule" "rds_postgres_from_ecs" {
  type                     = "ingress"
  description              = "Allow ECS tasks to access Postgres"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = aws_security_group.ecs_tasks.id
}

module "ecr" {
  source          = "./modules/ecr"
  repository_name = var.project_name
  force_delete    = true
  tags            = local.tags
}

module "s3" {
  source        = "./modules/s3"
  bucket_name   = "${var.project_name}-artifacts"
  force_destroy = true
  tags          = local.tags
}

module "secrets" {
  source         = "./modules/secrets"
  openai_api_key = var.openai_api_key
  name_prefix    = "/${var.project_name}"
  tags           = local.tags
}

module "sqs" {
  source = "./modules/sqs"

  name_prefix        = var.project_name
  visibility_timeout = var.sqs_visibility_timeout
  max_receive_count  = var.sqs_max_receive_count
  tags               = local.tags
}

module "rds" {
  source = "./modules/rds"

  name_prefix        = var.project_name
  subnet_ids         = var.subnet_ids
  security_group_ids = concat([aws_security_group.rds.id], local.rds_security_group_ids)
  instance_class     = var.db_instance_class
  db_username        = var.db_username
  db_password        = var.db_password
  tags               = local.tags
}

resource "aws_ssm_parameter" "rds_dsn" {
  name        = "/${var.project_name}/rds-dsn"
  description = "Postgres DSN for distributed ingestion"
  type        = "SecureString"
  value       = module.rds.dsn
  tags        = local.tags
}

module "efs" {
  source = "./modules/efs"
  count  = var.enable_qdrant_efs ? 1 : 0

  name_prefix        = var.project_name
  subnet_ids         = var.subnet_ids
  security_group_ids = [aws_security_group.ecs_tasks.id]
  vpc_id             = data.aws_subnet.first.vpc_id
  tags               = local.tags
}

module "ecs" {
  source = "./modules/ecs"

  cluster_name       = var.project_name
  app_image          = "${module.ecr.repository_url}:${var.app_image_tag}"
  openai_api_key_arn = module.secrets.openai_api_key_arn
  s3_bucket_arn      = module.s3.bucket_arn
  corpus_bucket_arn  = "arn:aws:s3:::${local.corpus_bucket_name}"
  subnet_ids         = var.subnet_ids
  security_group_ids = concat([aws_security_group.ecs_tasks.id], local.ecs_security_group_ids)

  app_desired_count    = var.app_desired_count
  qdrant_desired_count = var.qdrant_desired_count
  worker_desired_count = var.worker_desired_count
  worker_cpu           = var.worker_cpu
  worker_memory        = var.worker_memory
  sqs_queue_url        = module.sqs.queue_url
  sqs_queue_arn        = module.sqs.queue_arn
  s3_bucket_name       = local.corpus_bucket_name
  rds_dsn_arn          = aws_ssm_parameter.rds_dsn.arn
  corpus_s3_prefix     = var.corpus_s3_prefix
  chunk_s3_prefix      = var.chunk_s3_prefix
  chunk_max_s3_workers = var.chunk_max_s3_workers
  orchestrator_cpu     = var.orchestrator_cpu
  orchestrator_memory  = var.orchestrator_memory
  query_eval_cpu       = var.query_eval_cpu
  query_eval_memory    = var.query_eval_memory
  eval_s3_prefix       = var.eval_s3_prefix
  manifests_s3_prefix  = var.manifests_s3_prefix
  qdrant_collection    = var.qdrant_collection
  qdrant_efs_file_system_id  = var.enable_qdrant_efs ? module.efs[0].file_system_id : ""
  qdrant_efs_access_point_id = var.enable_qdrant_efs ? module.efs[0].access_point_id : ""

  tags = local.tags
}
