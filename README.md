# LLM MLOps Platform

> A production-grade MLOps platform for LLM fine-tuning, RAG, and serving on Kubernetes.

[![Terraform](https://img.shields.io/badge/Terraform-1.7+-623CE4?logo=terraform)](https://www.terraform.io/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.29+-326CE5?logo=kubernetes)](https://kubernetes.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![ArgoCD](https://img.shields.io/badge/ArgoCD-GitOps-EF7B4D?logo=argo)](https://argoproj.github.io/cd/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This platform provides end-to-end infrastructure for:

- **Data Processing**: Document ingestion, chunking, and embedding generation with Unstructured.io
- **Training**: Distributed LLM fine-tuning with LoRA/QLoRA and DeepSpeed ZeRO-3
- **Serving**: High-throughput inference with vLLM and KServe
- **RAG**: Retrieval-Augmented Generation with Milvus vector store
- **Observability**: Full metrics, logging, tracing, and alerting stack

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

## Key Features

### Infrastructure as Code
- **Terraform Modules**: Networking, EKS, IAM, Storage, Security, ECR
- **GitOps with ArgoCD**: App-of-apps pattern for declarative deployments
- **Multi-environment Support**: Dev, Staging, Production configurations

### ML Training
- **LoRA/QLoRA Fine-tuning**: Parameter-efficient training with PEFT
- **Distributed Training**: DeepSpeed ZeRO-3 and PyTorch Distributed
- **Hyperparameter Optimization**: Ray Tune integration with ASHA scheduler
- **Experiment Tracking**: MLflow for metrics, params, and model registry

### Model Serving
- **High-throughput Inference**: vLLM for fast LLM serving
- **Serverless Deployment**: KServe InferenceService
- **API Gateway**: LiteLLM for unified OpenAI-compatible API
- **Traffic Management**: Istio for canary deployments and A/B testing

### Observability
- **Metrics**: Prometheus + Grafana dashboards
- **Logging**: Loki for centralized log aggregation
- **Alerting**: Comprehensive Prometheus alerting rules
- **Tracing**: Distributed request tracing

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

### Useful Commands

```bash
# Development
make lint          # Run linting checks
make format        # Format code
make test          # Run unit tests
make test-all      # Run all tests

# Kubernetes
make port-forward-grafana   # Access Grafana on localhost:3000
make port-forward-mlflow    # Access MLflow on localhost:5000
make port-forward-argocd    # Access ArgoCD on localhost:8080

# Docker
make docker-build-all       # Build all container images
```

## Project Structure

```
├── terraform/          # Infrastructure as Code
│   ├── modules/        # Reusable Terraform modules
│   │   ├── networking/ # VPC, subnets, NAT gateways
│   │   ├── eks/        # EKS cluster, node groups
│   │   ├── iam/        # IAM roles and policies
│   │   ├── storage/    # S3 buckets, EFS
│   │   ├── security/   # Security groups, KMS
│   │   └── ecr/        # Container registries
│   ├── environments/   # Environment configurations
│   └── shared/         # Shared resources (state backend)
├── kubernetes/         # Kubernetes manifests
│   ├── argocd/         # ArgoCD app-of-apps
│   ├── base/           # Common resources
│   ├── helm-values/    # Helm chart values per environment
│   └── README.md       # Kubernetes architecture docs
├── pipelines/          # Kubeflow pipeline definitions
│   ├── data/           # Data ingestion pipeline
│   ├── training/       # Training pipeline
│   ├── evaluation/     # Model evaluation pipeline
│   └── serving/        # Model deployment pipeline
├── src/                # Python source code
│   ├── data/           # Data processing modules
│   ├── training/       # Training modules
│   ├── evaluation/     # Evaluation modules
│   ├── serving/        # Model serving modules
│   └── common/         # Shared utilities
├── tests/              # Test suites
│   ├── data/           # Data module tests
│   ├── training/       # Training tests
│   ├── serving/        # Serving tests
│   └── e2e/            # End-to-end tests
├── docker/             # Dockerfiles
├── docs/               # Documentation
│   ├── architecture/   # Architecture docs
│   ├── guides/         # User guides
│   └── runbooks/       # Operational runbooks
├── scripts/            # Automation scripts
└── configs/            # Configuration files
```


## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Cloud** | AWS (EKS, S3, RDS, ElastiCache, ECR) |
| **Infrastructure** | Terraform, Helm, Kustomize |
| **Orchestration** | Kubernetes 1.29+, ArgoCD |
| **Service Mesh** | Istio |
| **ML Training** | PyTorch, Transformers, PEFT, DeepSpeed |
| **HPO** | Ray Tune |
| **Pipeline** | Kubeflow Pipelines v2 |
| **Experiment Tracking** | MLflow |
| **Vector Store** | Milvus |
| **Data Processing** | Ray, Unstructured.io |
| **Model Serving** | vLLM, KServe |
| **API Gateway** | LiteLLM |
| **Monitoring** | Prometheus, Grafana, Loki |
| **Language** | Python 3.10+ |

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Getting Started](docs/guides/getting-started.md)
- [Kubernetes Architecture](kubernetes/README.md)
- [Runbooks](docs/runbooks/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run pre-commit hooks (`pre-commit run --all-files`)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - See [LICENSE](LICENSE) for details.
