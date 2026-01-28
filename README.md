# LLM MLOps Platform

> A production-grade MLOps platform for LLM fine-tuning, RAG, and serving on Kubernetes.

[![Terraform](https://img.shields.io/badge/Terraform-1.7+-623CE4?logo=terraform)](https://www.terraform.io/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.29+-326CE5?logo=kubernetes)](https://kubernetes.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)

## Overview

This platform provides end-to-end infrastructure for:
- **Data Processing**: Document ingestion, chunking, and embedding generation
- **Training**: Distributed LLM fine-tuning with LoRA/QLoRA
- **Serving**: High-throughput inference with vLLM and KServe
- **RAG**: Retrieval-Augmented Generation with Milvus vector store
- **Observability**: Full metrics, logging, and tracing stack

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           EKS Cluster                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   ArgoCD    │  │    Istio    │  │ Prometheus  │  │   Grafana   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   MLflow    │  │  Kubeflow   │  │     Ray     │  │   Milvus    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │    vLLM     │  │   KServe    │  │   LiteLLM   │                    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │    S3    │  │   ECR    │  │   RDS    │  │   KMS    │                     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Required tools
python >= 3.10
kubectl >= 1.29
helm >= 3.14
terraform >= 1.7
aws-cli >= 2.15
docker >= 24.0
```

### Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/llm-mlops-platform.git
cd llm-mlops-platform

# 2. Install dependencies
make install

# 3. Configure AWS
aws configure --profile llm-mlops-dev
export AWS_PROFILE=llm-mlops-dev

# 4. Initialize Terraform state (one-time)
cd terraform/shared/state-backend
terraform init && terraform apply
cd ../../..

# 5. Deploy infrastructure
make deploy-dev
```

## Project Structure

```
├── terraform/          # Infrastructure as Code
│   ├── modules/        # Reusable Terraform modules
│   ├── environments/   # Environment configurations
│   └── shared/         # Shared resources (state backend)
├── kubernetes/         # Kubernetes manifests
│   ├── argocd/         # ArgoCD app-of-apps
│   ├── base/           # Common resources
│   ├── components/     # Platform components
│   └── overlays/       # Environment overlays
├── pipelines/          # Kubeflow pipeline definitions
├── src/                # Python source code
├── tests/              # Test suites
├── docker/             # Dockerfiles
├── docs/               # Documentation
├── scripts/            # Automation scripts
└── configs/            # Configuration files
```

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Foundation | VPC, EKS, S3, IAM | 🔄 In Progress |
| 2. Platform Services | ArgoCD, Istio, Monitoring | ✅ Complete |
| 3. Data Infrastructure | Ray, Milvus, Redis | ✅ Complete |
| 4. Training Infrastructure | Kubeflow, MLflow | ✅ Complete |
| 5. Data Pipeline | Ingestion, Embedding | ✅ Complete |
| 6. Training Pipeline | LoRA Fine-tuning | ✅ Complete |
| 7. Serving Infrastructure | vLLM, KServe | ✅ Complete |
| 8. Integration | E2E Testing, Hardening | ⬜ Pending |

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Getting Started](docs/guides/getting-started.md)
- [Local Development](docs/guides/local-development.md)
- [Deployment Guide](docs/guides/deployment.md)
- [Runbooks](docs/runbooks/)

## License

MIT License - See [LICENSE](LICENSE) for details.
