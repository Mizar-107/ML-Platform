output "eks_kms_key_arn" {
  description = "ARN of the KMS key for EKS encryption"
  value       = aws_kms_key.eks.arn
}

output "eks_kms_key_id" {
  description = "ID of the KMS key for EKS encryption"
  value       = aws_kms_key.eks.key_id
}

output "ebs_kms_key_arn" {
  description = "ARN of the KMS key for EBS encryption"
  value       = aws_kms_key.ebs.arn
}

output "s3_kms_key_arn" {
  description = "ARN of the KMS key for S3 encryption"
  value       = aws_kms_key.s3.arn
}

output "eks_cluster_security_group_id" {
  description = "Security group ID for EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "eks_nodes_security_group_id" {
  description = "Security group ID for EKS nodes"
  value       = aws_security_group.eks_nodes.id
}

output "database_security_group_id" {
  description = "Security group ID for databases"
  value       = var.create_database_sg ? aws_security_group.database[0].id : null
}
