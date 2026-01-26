#!/bin/bash
# Bootstrap EKS cluster with all platform services
# Usage: ./bootstrap-cluster.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRY_RUN="${1:-}"

echo "=== Platform Bootstrap Script ==="
echo "Repo root: $REPO_ROOT"

# Check prerequisites
for cmd in kubectl helm aws; do
  command -v $cmd >/dev/null 2>&1 || { echo "$cmd is required but not installed. Aborting." >&2; exit 1; }
done

# Verify cluster connectivity
echo "==> Verifying cluster connectivity..."
kubectl cluster-info >/dev/null 2>&1 || { echo "Cannot connect to Kubernetes cluster. Check your kubeconfig." >&2; exit 1; }

# Step 1: Create namespaces
echo ""
echo "=== Step 1: Creating Namespaces ==="
kubectl apply -f "$REPO_ROOT/kubernetes/base/namespaces.yaml"

# Step 2: Deploy ArgoCD
echo ""
echo "=== Step 2: Deploying ArgoCD ==="
"$SCRIPT_DIR/deploy-argocd.sh"

# Step 3: Deploy app-of-apps to bootstrap all platform services
echo ""
echo "=== Step 3: Deploying Platform Applications ==="
if [ "$DRY_RUN" = "--dry-run" ]; then
  echo "[DRY RUN] Would apply: $REPO_ROOT/kubernetes/argocd/apps/app-of-apps.yaml"
else
  kubectl apply -f "$REPO_ROOT/kubernetes/argocd/apps/app-of-apps.yaml"
fi

# Step 4: Wait for ArgoCD to sync applications
echo ""
echo "=== Step 4: Waiting for Applications to Sync ==="
echo "ArgoCD will now deploy the following applications:"
echo "  - Istio (service mesh)"
echo "  - cert-manager (TLS certificates)"
echo "  - external-secrets (AWS Secrets Manager)"
echo "  - monitoring (Prometheus + Grafana)"
echo "  - loki (log aggregation)"
echo ""
echo "Monitor progress in ArgoCD UI:"
echo "  kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "  Open https://localhost:8080"

# Check app status
sleep 10
argocd app list 2>/dev/null || kubectl get applications -n argocd

# Step 5: Post-install instructions
echo ""
echo "=== Step 5: Post-Install Resources ==="
echo "After all applications are synced, apply these resources:"
echo "  kubectl apply -f $REPO_ROOT/kubernetes/base/cluster-issuers.yaml"
echo "  kubectl apply -f $REPO_ROOT/kubernetes/base/cluster-secret-stores.yaml"
echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo "Next steps:"
echo "1. Wait for ArgoCD to sync all applications (check UI)"
echo "2. Apply ClusterIssuers and ClusterSecretStores"
echo "3. Configure IRSA for external-secrets service account"
echo "4. Update GitHub repository URL in ArgoCD Applications"
