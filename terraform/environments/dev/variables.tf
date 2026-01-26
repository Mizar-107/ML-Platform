# General
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "llm-mlops"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones"
  type        = number
  default     = 3
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway (cost savings)"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = false
}

# EKS
variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "endpoint_public_access" {
  description = "Enable public API access"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "CIDRs for public API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# System Node Group
variable "system_node_instance_types" {
  description = "Instance types for system nodes"
  type        = list(string)
  default     = ["m6i.large"]
}

variable "system_node_desired_size" {
  description = "Desired system nodes"
  type        = number
  default     = 2
}

variable "system_node_min_size" {
  description = "Minimum system nodes"
  type        = number
  default     = 2
}

variable "system_node_max_size" {
  description = "Maximum system nodes"
  type        = number
  default     = 4
}

# GPU Node Group
variable "enable_gpu_nodes" {
  description = "Enable GPU nodes"
  type        = bool
  default     = true
}

variable "gpu_node_instance_types" {
  description = "GPU instance types"
  type        = list(string)
  default     = ["g5.xlarge"]
}

variable "gpu_capacity_type" {
  description = "GPU capacity type"
  type        = string
  default     = "SPOT"  # Use spot for dev
}

variable "gpu_node_max_size" {
  description = "Maximum GPU nodes"
  type        = number
  default     = 4
}

# Data Node Group
variable "enable_data_nodes" {
  description = "Enable data processing nodes"
  type        = bool
  default     = true
}

variable "data_node_instance_types" {
  description = "Data node instance types"
  type        = list(string)
  default     = ["r6i.2xlarge"]
}

variable "data_node_max_size" {
  description = "Maximum data nodes"
  type        = number
  default     = 10
}

# Database
variable "create_database" {
  description = "Create database security groups"
  type        = bool
  default     = false  # Disabled for dev to save costs
}

# Storage
variable "enable_versioning" {
  description = "Enable S3 versioning"
  type        = bool
  default     = true
}

variable "artifacts_retention_days" {
  description = "Artifacts retention days"
  type        = number
  default     = 365
}

variable "logs_retention_days" {
  description = "Logs retention days"
  type        = number
  default     = 90
}

variable "create_efs" {
  description = "Create EFS filesystem"
  type        = bool
  default     = false  # Disabled for dev to save costs
}
