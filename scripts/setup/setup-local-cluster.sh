#!/bin/bash
# Setup local development Kind cluster
# Usage: ./setup-local-cluster.sh

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-mlops-local}"
K8S_VERSION="${K8S_VERSION:-v1.29.0}"

echo "==> Setting up local Kind cluster: ${CLUSTER_NAME}"

# Check if kind is installed
if ! command -v kind &> /dev/null; then
    echo "Error: kind is not installed"
    echo "Install with: brew install kind (macOS) or see https://kind.sigs.k8s.io/"
    exit 1
fi

# Delete existing cluster if exists
kind delete cluster --name ${CLUSTER_NAME} 2>/dev/null || true

# Create cluster with config
cat <<EOF | kind create cluster --name ${CLUSTER_NAME} --image kindest/node:${K8S_VERSION} --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
  labels:
    node-type: cpu
    role: system
- role: worker
  labels:
    node-type: cpu
    role: data
EOF

# Wait for cluster to be ready
echo "==> Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s

# Install metrics server
echo "==> Installing metrics server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

echo ""
echo "==> Local cluster ${CLUSTER_NAME} is ready!"
echo ""
echo "Cluster nodes:"
kubectl get nodes
echo ""
echo "To use this cluster:"
echo "    kubectl config use-context kind-${CLUSTER_NAME}"
