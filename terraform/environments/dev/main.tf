################################################################################
# LLM MLOps Platform - Dev Environment
# Composes all modules for development deployment
################################################################################

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket         = "llm-mlops-terraform-state-us-west-2"
    key            = "dev/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "llm-mlops-terraform-locks"
    encrypt        = true
  }
}

################################################################################
# Provider Configuration
################################################################################

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

################################################################################
# Local Variables
################################################################################

locals {
  name_prefix  = "${var.project_name}-${var.environment}"
  cluster_name = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Repository  = "llm-mlops-platform"
  }
}

################################################################################
# Networking Module
################################################################################

module "networking" {
  source = "../../modules/networking"

  name_prefix        = local.name_prefix
  vpc_cidr           = var.vpc_cidr
  az_count           = var.az_count
  cluster_name       = local.cluster_name
  enable_nat_gateway = var.enable_nat_gateway
  single_nat_gateway = var.single_nat_gateway  # Cost savings for dev
  enable_vpc_endpoints = true
  enable_flow_logs   = var.enable_flow_logs

  tags = local.common_tags
}

################################################################################
# Security Module
################################################################################

module "security" {
  source = "../../modules/security"

  name_prefix        = local.name_prefix
  vpc_id             = module.networking.vpc_id
  cluster_name       = local.cluster_name
  create_database_sg = var.create_database

  tags = local.common_tags
}

################################################################################
# IAM Module (Initial - without OIDC)
################################################################################

module "iam" {
  source = "../../modules/iam"

  name_prefix     = local.name_prefix
  oidc_issuer_url = ""  # Will be updated after EKS is created

  tags = local.common_tags
}

################################################################################
# EKS Module
################################################################################

module "eks" {
  source = "../../modules/eks"

  cluster_name               = local.cluster_name
  cluster_version            = var.cluster_version
  cluster_role_arn           = module.iam.eks_cluster_role_arn
  node_role_arn              = module.iam.eks_node_role_arn
  subnet_ids                 = module.networking.private_subnet_ids
  cluster_security_group_ids = [module.security.eks_cluster_security_group_id]
  kms_key_arn                = module.security.eks_kms_key_arn
  endpoint_public_access     = var.endpoint_public_access
  public_access_cidrs        = var.public_access_cidrs

  # Enable EBS CSI with IRSA (after second apply)
  enable_ebs_csi   = true
  ebs_csi_role_arn = null  # Set after IRSA is configured

  # System nodes - always on
  system_node_instance_types = var.system_node_instance_types
  system_node_desired_size   = var.system_node_desired_size
  system_node_min_size       = var.system_node_min_size
  system_node_max_size       = var.system_node_max_size

  # GPU nodes - scale to zero for cost savings
  enable_gpu_nodes        = var.enable_gpu_nodes
  gpu_node_instance_types = var.gpu_node_instance_types
  gpu_capacity_type       = var.gpu_capacity_type
  gpu_node_desired_size   = 0  # Scale to zero when not in use
  gpu_node_min_size       = 0
  gpu_node_max_size       = var.gpu_node_max_size

  # Data processing nodes - spot instances
  enable_data_nodes        = var.enable_data_nodes
  data_node_instance_types = var.data_node_instance_types
  data_node_desired_size   = 0  # Scale to zero when not in use
  data_node_min_size       = 0
  data_node_max_size       = var.data_node_max_size

  tags = local.common_tags

  depends_on = [module.networking, module.security, module.iam]
}

################################################################################
# IAM Module (IRSA - Second Phase)
################################################################################

module "iam_irsa" {
  source = "../../modules/iam"

  name_prefix     = local.name_prefix
  oidc_issuer_url = module.eks.cluster_oidc_issuer_url

  tags = local.common_tags

  depends_on = [module.eks]
}

################################################################################
# Storage Module
################################################################################

module "storage" {
  source = "../../modules/storage"

  name_prefix              = var.project_name
  environment              = var.environment
  kms_key_arn              = module.security.s3_kms_key_arn
  enable_versioning        = var.enable_versioning
  artifacts_retention_days = var.artifacts_retention_days
  logs_retention_days      = var.logs_retention_days

  # EFS for shared storage (optional for dev)
  create_efs             = var.create_efs
  subnet_ids             = module.networking.private_subnet_ids
  efs_security_group_ids = [module.security.eks_nodes_security_group_id]

  tags = local.common_tags

  depends_on = [module.security]
}

################################################################################
# ECR Module
################################################################################

module "ecr" {
  source = "../../modules/ecr"

  name_prefix = var.project_name
  kms_key_arn = module.security.s3_kms_key_arn

  tags = local.common_tags

  depends_on = [module.security]
}
