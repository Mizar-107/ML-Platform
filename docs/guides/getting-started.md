# Getting Started Guide

Welcome to the LLM MLOps Platform! This guide will help you set up your development environment and run your first ML pipeline.

## Prerequisites

### Required Tools

Ensure you have the following tools installed:

```bash
# Check versions
python --version    # >= 3.10
kubectl version     # >= 1.29
helm version        # >= 3.14
terraform --version # >= 1.7
aws --version       # >= 2.15
docker --version    # >= 24.0
```

### AWS Configuration

1. Configure AWS credentials:

```bash
aws configure --profile llm-mlops-dev
export AWS_PROFILE=llm-mlops-dev
```

2. Verify access:

```bash
aws sts get-caller-identity
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/llm-mlops-platform.git
cd llm-mlops-platform
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your values
# Required: AWS_REGION, S3_BUCKET, MLFLOW_TRACKING_URI
```

### 4. Connect to Cluster

```bash
# Update kubeconfig (if cluster exists)
aws eks update-kubeconfig --name llm-mlops-dev --region us-west-2

# Verify connection
kubectl get nodes
```

## First Pipeline: Data Ingestion

Let's run a simple data ingestion pipeline:

### 1. Prepare Sample Data

```python
# scripts/prepare_sample_data.py
import json

sample_docs = [
    {
        "id": "doc1",
        "content": "MLOps combines Machine Learning and DevOps practices.",
        "metadata": {"source": "guide", "category": "mlops"}
    },
    {
        "id": "doc2", 
        "content": "LLMs are neural networks trained on text data.",
        "metadata": {"source": "guide", "category": "llm"}
    }
]

with open("data/sample_docs.json", "w") as f:
    json.dump(sample_docs, f, indent=2)

print(f"Created {len(sample_docs)} sample documents")
```

### 2. Run Data Pipeline

```bash
# Using the data module
python -m src.data.ingestion.loaders --input data/sample_docs.json --output data/processed/
```

### 3. Verify Results

```python
from src.data.storage.milvus import MilvusClient

client = MilvusClient()
count = client.get_collection_count("documents")
print(f"Vectors stored: {count}")
```

## Training Your First Model

### 1. Prepare Training Data

```bash
# Download sample training data
python scripts/download_sample_data.py --output data/training/
```

### 2. Configure Training

```yaml
# configs/training/lora-7b.yaml
model:
  name: "mistralai/Mistral-7B-v0.1"
  
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

training:
  learning_rate: 2e-4
  batch_size: 4
  num_epochs: 3
  max_seq_length: 512
```

### 3. Run Training (Local)

```bash
# Single GPU training
python -m src.training.trainers.lora_trainer \
  --config configs/training/lora-7b.yaml \
  --data_path data/training/ \
  --output_dir outputs/lora-7b/
```

### 4. Track with MLflow

```bash
# Start MLflow UI (if running locally)
mlflow ui --port 5000

# View at http://localhost:5000
```

## Deploying a Model

### 1. Register Model

```python
import mlflow

# Register trained model
model_uri = "outputs/lora-7b/final_model"
result = mlflow.register_model(model_uri, "mistral-7b-finetuned")
print(f"Registered model version: {result.version}")
```

### 2. Deploy to KServe

```bash
# Apply InferenceService
kubectl apply -f kubernetes/base/serving/inference-service.yaml

# Check status
kubectl get inferenceservice -n mlops-serving
```

### 3. Test Inference

```bash
curl -X POST http://mistral-7b.mlops-serving.example.com/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is MLOps?", "max_tokens": 100}'
```

## Running Evaluations

### RAGAS Evaluation

```python
from src.evaluation import RAGASEvaluator, RAGSample

evaluator = RAGASEvaluator()

samples = [
    RAGSample(
        question="What is MLOps?",
        answer="MLOps combines ML and DevOps.",
        contexts=["MLOps is the practice of..."]
    )
]

result = evaluator.evaluate_sync(samples)
print(f"Scores: {result.scores}")
```

### DeepEval Evaluation

```python
from src.evaluation import DeepEvalEvaluator, LLMTestCase

evaluator = DeepEvalEvaluator()

cases = [
    LLMTestCase(
        input="Explain machine learning",
        actual_output="Machine learning is...",
    )
]

result = evaluator.evaluate_sync(cases)
print(f"Pass rate: {result.pass_rate:.2%}")
```

## Running Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run E2E tests (requires cluster)
pytest tests/e2e/ -v -m "not requires_cluster"

# Run with coverage
pytest --cov=src --cov-report=html
```

## Common Tasks

### Port Forwarding

```bash
# MLflow UI
kubectl port-forward svc/mlflow -n mlflow 5000:5000

# Grafana
kubectl port-forward svc/grafana -n monitoring 3000:80

# Kubeflow Pipelines
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

### Viewing Logs

```bash
# Training job logs
kubectl logs -f job/training-job-xxx -n mlops-training

# Serving logs
kubectl logs -f deployment/mistral-7b-predictor -n mlops-serving

# ArgoCD logs
kubectl logs -f deployment/argocd-server -n argocd
```

### Scaling Services

```bash
# Scale inference service
kubectl patch inferenceservice mistral-7b -n mlops-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/predictor/minReplicas", "value": 2}]'
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Cluster not found" | Run `aws eks update-kubeconfig` |
| "Permission denied" | Check IRSA role or kubeconfig |
| "OOM during training" | Reduce batch size or enable gradient checkpointing |
| "Model not loading" | Check S3 permissions and model path |

### Getting Help

- Check [runbooks](../runbooks/) for operational procedures
- Review [architecture docs](../architecture/) for system design
- Open an issue on GitHub for bugs

## Next Steps

1. **Explore Pipelines**: Check out `pipelines/` for more examples
2. **Customize Training**: Modify configs in `configs/training/`
3. **Add Monitoring**: Set up dashboards in Grafana
4. **Production Deployment**: Follow the deployment guide
