#!/bin/bash
# Deploy ArgoCD to EKS cluster
# Usage: ./deploy-argocd.sh

set -euo pipefail

NAMESPACE="argocd"
ARGOCD_VERSION="${ARGOCD_VERSION:-v2.9.3}"

echo "==> Deploying ArgoCD ${ARGOCD_VERSION}..."

# Create namespace
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Install ArgoCD
kubectl apply -n ${NAMESPACE} -f https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml

# Wait for ArgoCD to be ready
echo "==> Waiting for ArgoCD to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n ${NAMESPACE}

# Get initial admin password
echo "==> ArgoCD Initial Admin Password:"
kubectl -n ${NAMESPACE} get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

# Port forward instructions
echo ""
echo "==> To access ArgoCD UI:"
echo "    kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "    Open https://localhost:8080"
echo "    Username: admin"
echo ""

echo "==> ArgoCD deployment complete!"
