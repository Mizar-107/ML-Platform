#!/bin/bash
# Deploy ArgoCD to EKS cluster using Helm
# Usage: ./deploy-argocd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARGOCD_NAMESPACE="argocd"
ARGOCD_VERSION="${ARGOCD_VERSION:-7.3.0}"

echo "=== ArgoCD Bootstrap Script ==="
echo "Repo root: $REPO_ROOT"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm is required but not installed. Aborting." >&2; exit 1; }

# Verify cluster connectivity
echo "==> Verifying cluster connectivity..."
kubectl cluster-info >/dev/null 2>&1 || { echo "Cannot connect to Kubernetes cluster. Check your kubeconfig." >&2; exit 1; }

# Apply namespaces
echo "==> Creating namespaces..."
kubectl apply -f "$REPO_ROOT/kubernetes/base/namespaces.yaml"

# Add Argo Helm repo
echo "==> Adding Argo Helm repository..."
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

# Install ArgoCD using Helm
echo "==> Installing ArgoCD ${ARGOCD_VERSION}..."
helm upgrade --install argocd argo/argo-cd \
  --namespace "$ARGOCD_NAMESPACE" \
  --version "$ARGOCD_VERSION" \
  --values "$REPO_ROOT/kubernetes/argocd/install/values.yaml" \
  --wait \
  --timeout 10m

# Wait for ArgoCD to be ready
echo "==> Waiting for ArgoCD to be ready..."
kubectl wait --for=condition=available deployment/argocd-server -n "$ARGOCD_NAMESPACE" --timeout=300s

# Get initial admin password
echo ""
echo "=== ArgoCD Installed Successfully ==="
echo ""
echo "==> ArgoCD Initial Admin Password:"
kubectl get secret argocd-initial-admin-secret -n "$ARGOCD_NAMESPACE" -o jsonpath="{.data.password}" | base64 -d
echo ""
echo ""
echo "==> To access ArgoCD UI:"
echo "    kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "    Open https://localhost:8080"
echo "    Username: admin"
echo ""
echo "==> To deploy platform apps, run:"
echo "    kubectl apply -f $REPO_ROOT/kubernetes/argocd/apps/app-of-apps.yaml"
echo ""
echo "==> ArgoCD deployment complete!"
