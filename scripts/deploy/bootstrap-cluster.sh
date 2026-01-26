#!/bin/bash
# Bootstrap cluster with ArgoCD apps
# Usage: ./bootstrap-cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Bootstrapping cluster with ArgoCD apps..."

# Apply app-of-apps
kubectl apply -f ${REPO_ROOT}/kubernetes/argocd/apps/app-of-apps.yaml

# Wait for sync
echo "==> Waiting for apps to sync..."
sleep 30

# Check app status
argocd app list 2>/dev/null || kubectl get applications -n argocd

echo "==> Bootstrap complete!"
echo ""
echo "Monitor apps with:"
echo "    argocd app list"
echo "    kubectl get applications -n argocd"
