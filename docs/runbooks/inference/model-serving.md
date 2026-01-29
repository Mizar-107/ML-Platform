# Model Serving Runbook

> **Last Updated:** 2026-01-29  
> **Owner:** ML Platform Team  
> **Severity Levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

## Overview

This runbook covers operational procedures for model serving infrastructure including KServe, vLLM, and LiteLLM gateway management.

---

## Quick Reference

| Action | Command |
|--------|---------|
| List InferenceServices | `kubectl get inferenceservice -n mlops-serving` |
| Check model status | `kubectl describe isvc <name> -n mlops-serving` |
| View serving logs | `kubectl logs -f deploy/<model>-predictor -n mlops-serving` |
| Test endpoint | `curl -X POST <url>/v1/completions -d '{"prompt":"test"}'` |

---

## Common Operations

### 1. Deploying a New Model

**Prerequisites:**
- Model registered in MLflow Model Registry
- Model artifacts in S3

```bash
# 1. Create InferenceService manifest
cat <<EOF > inference-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: mistral-7b-v2
  namespace: mlops-serving
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3
    model:
      modelFormat:
        name: vLLM
      storageUri: s3://llm-mlops-models/mistral-7b/v2
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: "24Gi"
        requests:
          memory: "16Gi"
EOF

# 2. Apply manifest
kubectl apply -f inference-service.yaml

# 3. Monitor deployment
kubectl get isvc mistral-7b-v2 -n mlops-serving -w

# 4. Wait for Ready status (typically 2-5 minutes for model loading)
kubectl wait --for=condition=Ready isvc/mistral-7b-v2 -n mlops-serving --timeout=600s
```

---

### 2. Updating a Model Version

**When to use:** Deploying new model version with traffic control

#### Blue-Green Deployment

```bash
# 1. Deploy new version alongside existing
kubectl apply -f inference-service-v2.yaml

# 2. Verify new version is ready
kubectl get isvc mistral-7b-v2 -n mlops-serving

# 3. Update LiteLLM to route to new version
kubectl patch configmap litellm-config -n mlops-serving \
  --type='json' -p='[
    {"op": "replace", "path": "/data/model_list/0/litellm_params/api_base", 
     "value": "http://mistral-7b-v2-predictor.mlops-serving.svc.cluster.local/v1"}
  ]'

# 4. Restart LiteLLM to pick up config
kubectl rollout restart deployment/litellm -n mlops-serving

# 5. Verify traffic flows to new version
kubectl logs -f deployment/litellm -n mlops-serving | grep "mistral-7b-v2"
```

#### Canary Deployment

```bash
# 1. Deploy canary with traffic split
kubectl apply -f - <<EOF
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: mistral-7b
  namespace: mlops-serving
spec:
  predictor:
    canaryTrafficPercent: 10
    model:
      modelFormat:
        name: vLLM
      storageUri: s3://llm-mlops-models/mistral-7b/v2
EOF

# 2. Monitor canary metrics
# Check Grafana dashboard: Serving Metrics > Canary Comparison

# 3. Gradually increase traffic
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/predictor/canaryTrafficPercent", "value": 50}]'

# 4. Complete promotion (100% to new version)
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/predictor/canaryTrafficPercent", "value": 100}]'
```

---

### 3. Scaling Inference Services

#### Manual Scaling

```bash
# Scale up replicas
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[
    {"op": "replace", "path": "/spec/predictor/minReplicas", "value": 2},
    {"op": "replace", "path": "/spec/predictor/maxReplicas", "value": 5}
  ]'

# Scale down (during low traffic)
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[
    {"op": "replace", "path": "/spec/predictor/minReplicas", "value": 1}
  ]'
```

#### Configure Autoscaling

```yaml
# HPA based on CPU/Memory
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mistral-7b-hpa
  namespace: mlops-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mistral-7b-predictor
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

### 4. Rollback Procedures

**When to use:** New model version causing errors, performance degradation

```bash
# 1. Check current and previous revisions
kubectl get isvc mistral-7b -n mlops-serving -o yaml | grep -A5 status

# 2. Quick rollback - scale canary to 0
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/predictor/canaryTrafficPercent", "value": 0}]'

# 3. Full rollback - revert to previous storage URI
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[
    {"op": "replace", "path": "/spec/predictor/model/storageUri", "value": "s3://llm-mlops-models/mistral-7b/v1"}
  ]'

# 4. Verify rollback
kubectl get isvc mistral-7b -n mlops-serving
curl -X POST http://mistral-7b.mlops-serving.example.com/v1/completions \
  -d '{"prompt": "test", "max_tokens": 10}'
```

---

## Incident Response

### High Latency (P99 > 2s)

**Severity:** P1  
**Alert:** `ModelServingHighLatency`

```bash
# 1. Check current latency
kubectl exec -it deployment/prometheus -n monitoring -- \
  promql 'histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))'

# 2. Check if replicas are overloaded
kubectl top pods -n mlops-serving -l serving.kserve.io/inferenceservice=mistral-7b

# 3. Check GPU memory
kubectl exec -it <pod> -n mlops-serving -- nvidia-smi

# 4. Scale up if needed
kubectl patch isvc mistral-7b -n mlops-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/predictor/minReplicas", "value": 3}]'

# 5. Check for long prompts causing slow responses
kubectl logs deployment/mistral-7b-predictor -n mlops-serving | grep "tokens"
```

### High Error Rate (> 1%)

**Severity:** P1  
**Alert:** `ModelServingHighErrorRate`

```bash
# 1. Check error logs
kubectl logs deployment/mistral-7b-predictor -n mlops-serving --tail=100 | grep -i error

# 2. Common errors and solutions:
# - OOM: Reduce batch size or max_tokens
# - CUDA OOM: Reduce concurrent requests, increase GPU memory
# - Model loading: Check S3 permissions, storage URI

# 3. Check if specific requests are failing
kubectl logs deployment/litellm -n mlops-serving | grep -i "500\|502\|503"

# 4. Restart pods if transient error
kubectl rollout restart deployment/mistral-7b-predictor -n mlops-serving

# 5. If model is corrupted, redeploy
kubectl delete isvc mistral-7b -n mlops-serving
kubectl apply -f inference-service.yaml
```

### Model Not Loading

**Severity:** P1  
**Symptoms:** Pod in CrashLoopBackOff, Init containers failing

```bash
# 1. Check pod status
kubectl describe pod <pod-name> -n mlops-serving

# 2. Check init container logs (model download)
kubectl logs <pod-name> -n mlops-serving -c storage-initializer

# 3. Common issues:
# - S3 access denied: Check IRSA role
# - Model not found: Verify storageUri
# - Disk full: Check node storage

# 4. Verify S3 access
kubectl run --rm -it s3-test --image=amazon/aws-cli -- \
  s3 ls s3://llm-mlops-models/mistral-7b/

# 5. Check IRSA configuration
kubectl describe sa kserve-controller-manager -n kserve
```

### GPU Not Available

**Severity:** P1  
**Symptoms:** Pod stuck in Pending, `nvidia.com/gpu` insufficient

```bash
# 1. Check GPU availability
kubectl describe nodes | grep -A5 "Allocated resources"

# 2. Check GPU node status
kubectl get nodes -l nvidia.com/gpu.present=true

# 3. Scale GPU node group if needed
eksctl scale nodegroup --cluster llm-mlops-dev --name gpu-inference --nodes 3

# 4. Check if other pods are hoarding GPUs
kubectl get pods --all-namespaces -o wide | grep -i gpu

# 5. Check node taints
kubectl describe node <gpu-node> | grep Taints
```

---

## Performance Tuning

### vLLM Configuration

```yaml
# Optimized vLLM settings for production
env:
  - name: VLLM_TENSOR_PARALLEL_SIZE
    value: "1"
  - name: VLLM_MAX_MODEL_LEN
    value: "4096"
  - name: VLLM_GPU_MEMORY_UTILIZATION
    value: "0.9"
  - name: VLLM_MAX_NUM_BATCHED_TOKENS
    value: "8192"
  - name: VLLM_MAX_NUM_SEQS
    value: "256"
```

### LiteLLM Configuration

```yaml
# Rate limiting and caching
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: redis.mlops-data.svc.cluster.local
    ttl: 3600
  
  # Rate limits per API key
  general_settings:
    max_budget: 100.0
    budget_duration: 1d
```

---

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| P50 Latency | < 500ms | > 1s |
| P99 Latency | < 2s | > 5s |
| Error Rate | < 0.1% | > 1% |
| GPU Utilization | 60-80% | > 95% |
| Requests/sec | - | Based on capacity |

### Grafana Dashboards

- **Serving Overview**: `/d/serving-overview`
- **Model Performance**: `/d/model-performance`
- **GPU Metrics**: `/d/gpu-metrics`

---

## Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| ML Platform On-Call | PagerDuty | Immediate |
| Model Owner | Slack #model-team | 15 min |
| Infrastructure | Slack #platform | 30 min |
