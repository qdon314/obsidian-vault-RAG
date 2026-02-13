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
  corpus_bucket_name = var.corpus_bucket_name != "" ? var.corpus_bucket_name : module.s3.bucket_name
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
  security_group_ids = var.security_group_ids
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

module "ecs" {
  source = "./modules/ecs"

  cluster_name       = var.project_name
  app_image          = "${module.ecr.repository_url}:latest"
  openai_api_key_arn = module.secrets.openai_api_key_arn
  s3_bucket_arn      = module.s3.bucket_arn
  subnet_ids         = var.subnet_ids
  security_group_ids = var.security_group_ids

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

  tags = local.tags
}
