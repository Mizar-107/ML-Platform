# Terraform State Backend

This module creates the S3 bucket and DynamoDB table for Terraform state locking.

## Usage

```bash
cd terraform/shared/state-backend
terraform init
terraform apply
```

## Resources Created

- S3 bucket for state storage with versioning
- DynamoDB table for state locking
- KMS key for encryption
