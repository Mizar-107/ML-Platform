variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "kms_key_arn" {
  description = "ARN of KMS key for encryption"
  type        = string
}

variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "artifacts_retention_days" {
  description = "Days to retain artifacts before deletion"
  type        = number
  default     = 730  # 2 years
}

variable "logs_retention_days" {
  description = "Days to retain logs before deletion"
  type        = number
  default     = 365
}

# EFS Configuration
variable "create_efs" {
  description = "Create EFS filesystem for shared storage"
  type        = bool
  default     = false
}

variable "subnet_ids" {
  description = "Subnet IDs for EFS mount targets"
  type        = list(string)
  default     = []
}

variable "efs_security_group_ids" {
  description = "Security group IDs for EFS mount targets"
  type        = list(string)
  default     = []
}

variable "efs_throughput_mode" {
  description = "EFS throughput mode (bursting or provisioned)"
  type        = string
  default     = "bursting"
}

variable "efs_provisioned_throughput" {
  description = "Provisioned throughput in MiB/s (when throughput_mode is provisioned)"
  type        = number
  default     = 100
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
