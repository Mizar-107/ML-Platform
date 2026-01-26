################################################################################
# Dev Environment - Terraform Variables
# These values are optimized for development (cost-saving defaults)
################################################################################

project_name = "llm-mlops"
environment  = "dev"
aws_region   = "us-west-2"

# Networking - single NAT for cost savings
vpc_cidr           = "10.0.0.0/16"
az_count           = 3
enable_nat_gateway = true
single_nat_gateway = true
enable_flow_logs   = false

# EKS Configuration
cluster_version        = "1.29"
endpoint_public_access = true
public_access_cidrs    = ["0.0.0.0/0"]  # Restrict in production

# System Nodes - minimal for dev
system_node_instance_types = ["m6i.large"]
system_node_desired_size   = 2
system_node_min_size       = 2
system_node_max_size       = 4

# GPU Nodes - spot instances, scale to zero
enable_gpu_nodes        = true
gpu_node_instance_types = ["g5.xlarge"]
gpu_capacity_type       = "SPOT"
gpu_node_max_size       = 4

# Data Nodes - spot instances, scale to zero
enable_data_nodes        = true
data_node_instance_types = ["r6i.2xlarge"]
data_node_max_size       = 10

# Database - disabled for dev
create_database = false

# Storage
enable_versioning        = true
artifacts_retention_days = 365
logs_retention_days      = 90
create_efs               = false
