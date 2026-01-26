# Phase 2: Platform Services

Kubernetes resources for platform services deployed via Helm charts and ArgoCD.

## Pre-Deployment Configuration

> **⚠️ IMPORTANT:** Complete these steps before deploying to your cluster.

### 1. Update GitHub Repository URL

Update `YOUR_ORG` in all ArgoCD Application files:

```bash
# Files to update:
# - argocd/apps/app-of-apps.yaml
# - argocd/apps/platform/*.yaml

# Replace with your actual repository URL
sed -i 's|YOUR_ORG/llm-mlops-platform|your-org/your-repo|g' argocd/apps/*.yaml argocd/apps/platform/*.yaml
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

### 5. Change Grafana Admin Password

Update the default password in `helm-values/dev/kube-prometheus-stack.yaml`:

```yaml
grafana:
  adminPassword: your-secure-password  # Change from 'admin'
```

---

## Deployment

```bash
# Bootstrap everything
./scripts/deploy/bootstrap-cluster.sh

# Or step by step:
./scripts/deploy/deploy-argocd.sh
kubectl apply -f argocd/apps/app-of-apps.yaml

# After apps sync:
kubectl apply -f base/cluster-issuers.yaml
kubectl apply -f base/cluster-secret-stores.yaml
```

## Directory Structure

```
kubernetes/
├── argocd/
│   ├── install/values.yaml      # ArgoCD Helm values
│   └── apps/
│       ├── app-of-apps.yaml     # Root Application
│       └── platform/            # Platform service Applications
├── helm-values/
│   └── dev/                     # Dev environment Helm values
└── base/                        # Base resources (namespaces, issuers)
```

## Services Deployed

| Service | Namespace | Purpose |
|---------|-----------|---------|
| ArgoCD | argocd | GitOps controller |
| Istio | istio-system | Service mesh |
| Prometheus/Grafana | monitoring | Metrics & dashboards |
| Loki | monitoring | Log aggregation |
| cert-manager | cert-manager | TLS certificates |
| External Secrets | external-secrets | AWS Secrets Manager |
