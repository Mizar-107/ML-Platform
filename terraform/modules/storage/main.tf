################################################################################
# Storage Module
# Creates S3 buckets for data, models, and artifacts
################################################################################

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }
}

################################################################################
# S3 Buckets
################################################################################

# Data bucket - raw and processed training data
resource "aws_s3_bucket" "data" {
  bucket = "${var.name_prefix}-data-${var.environment}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-data"
    Purpose = "training-data"
  })
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Models bucket - trained model weights and checkpoints
resource "aws_s3_bucket" "models" {
  bucket = "${var.name_prefix}-models-${var.environment}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-models"
    Purpose = "model-artifacts"
  })
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"  # Always version models
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# Artifacts bucket - MLflow artifacts, pipeline outputs
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.name_prefix}-artifacts-${var.environment}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-artifacts"
    Purpose = "mlflow-artifacts"
  })
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = var.artifacts_retention_days
    }
  }
}

# Logs bucket - application and infrastructure logs
resource "aws_s3_bucket" "logs" {
  bucket = "${var.name_prefix}-logs-${var.environment}"

  tags = merge(var.tags, {
    Name    = "${var.name_prefix}-logs"
    Purpose = "logging"
  })
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id

  versioning_configuration {
    status = "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"  # Use S3-managed keys for logs
    }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "log-retention"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.logs_retention_days
    }
  }
}

################################################################################
# EFS Filesystem (for shared storage)
################################################################################

resource "aws_efs_file_system" "shared" {
  count = var.create_efs ? 1 : 0

  creation_token = "${var.name_prefix}-shared-efs"
  encrypted      = true
  kms_key_id     = var.kms_key_arn

  performance_mode                = "generalPurpose"
  throughput_mode                 = var.efs_throughput_mode
  provisioned_throughput_in_mibps = var.efs_throughput_mode == "provisioned" ? var.efs_provisioned_throughput : null

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-shared-efs"
  })
}

resource "aws_efs_mount_target" "shared" {
  count = var.create_efs ? length(var.subnet_ids) : 0

  file_system_id  = aws_efs_file_system.shared[0].id
  subnet_id       = var.subnet_ids[count.index]
  security_groups = var.efs_security_group_ids

  depends_on = [aws_efs_file_system.shared]
}

resource "aws_efs_access_point" "training" {
  count = var.create_efs ? 1 : 0

  file_system_id = aws_efs_file_system.shared[0].id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/training"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-training-ap"
  })
}
