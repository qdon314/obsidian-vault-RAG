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

  tags = local.tags
}
