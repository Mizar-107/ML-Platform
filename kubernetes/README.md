# Kubernetes Platform Services

Kubernetes resources for platform services and data infrastructure deployed via Helm charts and ArgoCD.

## Pre-Deployment Configuration

> **⚠️ IMPORTANT:** Complete these steps before deploying to your cluster.

### 1. Update GitHub Repository URL

Update `YOUR_ORG` in all ArgoCD Application files:

```bash
# Files to update:
# - argocd/apps/app-of-apps.yaml
# - argocd/apps/platform/*.yaml
# - argocd/apps/mlops-apps.yaml
# - argocd/apps/mlops/*.yaml

# Replace with your actual repository URL
sed -i 's|YOUR_ORG/llm-mlops-platform|your-org/your-repo|g' argocd/apps/*.yaml argocd/apps/platform/*.yaml argocd/apps/mlops/*.yaml
```

### 2. Configure AWS Region

Update the AWS region in `base/cluster-secret-stores.yaml`:

```yaml
region: us-west-2  # Change to your region
```

### 3. Configure Let's Encrypt Email

Update the email address in `base/cluster-issuers.yaml`:

```yaml
email: admin@example.com  # Change to your email
```

### 4. Configure IRSA for External Secrets

After deployment, annotate the service account with your IAM role:

```yaml
# In helm-values/dev/external-secrets.yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::YOUR_ACCOUNT_ID:role/external-secrets-role
```

### 5. Change Default Passwords

Update the default passwords in helm-values:

```yaml
# helm-values/dev/kube-prometheus-stack.yaml
grafana:
  adminPassword: your-secure-password

# helm-values/dev/redis.yaml
auth:
  password: your-redis-password

# helm-values/dev/milvus.yaml
minio:
  accessKey: your-minio-access-key
  secretKey: your-minio-secret-key
```

---

## Deployment

```bash
# Bootstrap everything
./scripts/deploy/bootstrap-cluster.sh

# Or step by step:
./scripts/deploy/deploy-argocd.sh
kubectl apply -f argocd/apps/app-of-apps.yaml

# After platform apps sync:
kubectl apply -f base/cluster-issuers.yaml
kubectl apply -f base/cluster-secret-stores.yaml

# Deploy MLOps/Data Infrastructure apps:
kubectl apply -f argocd/apps/mlops-apps.yaml
```

## Directory Structure

```
kubernetes/
├── argocd/
│   ├── install/values.yaml      # ArgoCD Helm values
│   └── apps/
│       ├── app-of-apps.yaml     # Root Application (Platform)
│       ├── mlops-apps.yaml      # Root Application (MLOps/Data)
│       ├── platform/            # Platform service Applications
│       └── mlops/               # MLOps/Data Infrastructure Applications
├── helm-values/
│   └── dev/                     # Dev environment Helm values
└── base/
    └── namespaces/              # Namespace definitions
```

## Services Deployed

### Phase 2: Platform Services

| Service | Namespace | Purpose |
|---------|-----------|---------|
| ArgoCD | argocd | GitOps controller |
| Istio | istio-system | Service mesh |
| Prometheus/Grafana | monitoring | Metrics & dashboards |
| Loki | monitoring | Log aggregation |
| cert-manager | cert-manager | TLS certificates |
| External Secrets | external-secrets | AWS Secrets Manager |

### Phase 3: Data Infrastructure

| Service | Namespace | Purpose |
|---------|-----------|---------|
| KubeRay Operator | kuberay-system | Ray cluster operator |
| Ray Cluster | ray-system | Distributed computing |
| Milvus | milvus | Vector database |
| Redis | data | Caching layer |

### Phase 4: Training Infrastructure

| Service | Namespace | Purpose |
|---------|-----------|---------|
| MLflow | mlflow | Experiment tracking & model registry |
| Training Operator | training-operator-system | PyTorchJob/TFJob CRDs |
| Volcano | volcano-system | Gang scheduling |
| Kubeflow Pipelines | kubeflow | ML workflow orchestration |

## Sync-Wave Order

ArgoCD sync-waves ensure dependencies deploy in order:

| Wave | Component |
|------|-----------|
| 1 | cert-manager |
| 2 | Istio |
| 3 | Monitoring (Prometheus/Grafana) |
| 4 | Loki, External Secrets |
| 5 | MLOps Apps (parent) |
| 10 | KubeRay Operator |
| 11 | Ray Cluster |
| 12 | Milvus, Redis |
| 13 | MLflow |
| 14 | Training Operator, Volcano |
| 15 | Kubeflow Pipelines |


