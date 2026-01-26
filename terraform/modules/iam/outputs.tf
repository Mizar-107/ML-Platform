output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster.arn
}

output "eks_node_role_arn" {
  description = "ARN of the EKS node IAM role"
  value       = aws_iam_role.eks_node.arn
}

output "oidc_provider_arn" {
  description = "ARN of the OIDC provider"
  value       = var.oidc_issuer_url != "" ? aws_iam_openid_connect_provider.eks[0].arn : null
}

output "mlflow_role_arn" {
  description = "ARN of the MLflow IRSA role"
  value       = var.oidc_issuer_url != "" ? aws_iam_role.mlflow[0].arn : null
}

output "training_role_arn" {
  description = "ARN of the training IRSA role"
  value       = var.oidc_issuer_url != "" ? aws_iam_role.training[0].arn : null
}

output "serving_role_arn" {
  description = "ARN of the serving IRSA role"
  value       = var.oidc_issuer_url != "" ? aws_iam_role.serving[0].arn : null
}

output "ebs_csi_role_arn" {
  description = "ARN of the EBS CSI driver IRSA role"
  value       = var.oidc_issuer_url != "" ? aws_iam_role.ebs_csi[0].arn : null
}
