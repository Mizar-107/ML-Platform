# Cluster Outputs
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Cluster CA certificate (base64)"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL"
  value       = module.eks.cluster_oidc_issuer_url
}

# Networking Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.networking.private_subnet_ids
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.networking.public_subnet_ids
}

# Storage Outputs
output "data_bucket_name" {
  description = "Data S3 bucket name"
  value       = module.storage.data_bucket_name
}

output "models_bucket_name" {
  description = "Models S3 bucket name"
  value       = module.storage.models_bucket_name
}

output "artifacts_bucket_name" {
  description = "Artifacts S3 bucket name"
  value       = module.storage.artifacts_bucket_name
}

# ECR Outputs
output "ecr_repository_urls" {
  description = "ECR repository URLs"
  value       = module.ecr.all_repository_urls
}

# IRSA Role Outputs
output "mlflow_role_arn" {
  description = "MLflow IRSA role ARN"
  value       = module.iam_irsa.mlflow_role_arn
}

output "training_role_arn" {
  description = "Training IRSA role ARN"
  value       = module.iam_irsa.training_role_arn
}

output "serving_role_arn" {
  description = "Serving IRSA role ARN"
  value       = module.iam_irsa.serving_role_arn
}

# Kubeconfig Command
output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}
