output "data_bucket_name" {
  description = "Name of the data S3 bucket"
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the data S3 bucket"
  value       = aws_s3_bucket.data.arn
}

output "models_bucket_name" {
  description = "Name of the models S3 bucket"
  value       = aws_s3_bucket.models.id
}

output "models_bucket_arn" {
  description = "ARN of the models S3 bucket"
  value       = aws_s3_bucket.models.arn
}

output "artifacts_bucket_name" {
  description = "Name of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.id
}

output "artifacts_bucket_arn" {
  description = "ARN of the artifacts S3 bucket"
  value       = aws_s3_bucket.artifacts.arn
}

output "logs_bucket_name" {
  description = "Name of the logs S3 bucket"
  value       = aws_s3_bucket.logs.id
}

output "logs_bucket_arn" {
  description = "ARN of the logs S3 bucket"
  value       = aws_s3_bucket.logs.arn
}

output "efs_id" {
  description = "ID of the EFS filesystem"
  value       = var.create_efs ? aws_efs_file_system.shared[0].id : null
}

output "efs_dns_name" {
  description = "DNS name of the EFS filesystem"
  value       = var.create_efs ? aws_efs_file_system.shared[0].dns_name : null
}

output "efs_training_access_point_id" {
  description = "ID of the training EFS access point"
  value       = var.create_efs ? aws_efs_access_point.training[0].id : null
}
