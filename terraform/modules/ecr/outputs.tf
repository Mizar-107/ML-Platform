output "training_repository_url" {
  description = "URL of the training ECR repository"
  value       = aws_ecr_repository.training.repository_url
}

output "training_repository_arn" {
  description = "ARN of the training ECR repository"
  value       = aws_ecr_repository.training.arn
}

output "serving_repository_url" {
  description = "URL of the serving ECR repository"
  value       = aws_ecr_repository.serving.repository_url
}

output "serving_repository_arn" {
  description = "ARN of the serving ECR repository"
  value       = aws_ecr_repository.serving.arn
}

output "pipeline_components_repository_url" {
  description = "URL of the pipeline components ECR repository"
  value       = aws_ecr_repository.pipeline_components.repository_url
}

output "pipeline_components_repository_arn" {
  description = "ARN of the pipeline components ECR repository"
  value       = aws_ecr_repository.pipeline_components.arn
}

output "data_repository_url" {
  description = "URL of the data processing ECR repository"
  value       = aws_ecr_repository.data.repository_url
}

output "data_repository_arn" {
  description = "ARN of the data processing ECR repository"
  value       = aws_ecr_repository.data.arn
}

output "all_repository_urls" {
  description = "Map of all ECR repository URLs"
  value = {
    training           = aws_ecr_repository.training.repository_url
    serving            = aws_ecr_repository.serving.repository_url
    pipeline_components = aws_ecr_repository.pipeline_components.repository_url
    data               = aws_ecr_repository.data.repository_url
  }
}
