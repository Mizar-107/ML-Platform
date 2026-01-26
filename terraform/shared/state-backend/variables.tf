variable "aws_region" {
  description = "AWS region for state backend"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "llm-mlops"
}
