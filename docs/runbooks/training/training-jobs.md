# Training Jobs Runbook

> **Last Updated:** 2026-01-29  
> **Owner:** ML Platform Team  
> **Severity Levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

## Overview

This runbook covers operational procedures for managing training jobs including PyTorchJob management, GPU allocation, checkpointing, and failure recovery.

---

## Quick Reference

| Action | Command |
|--------|---------|
| List training jobs | `kubectl get pytorchjob -n mlops-training` |
| Check job status | `kubectl describe pytorchjob <name> -n mlops-training` |
| View training logs | `kubectl logs -f <job>-master-0 -n mlops-training` |
| Cancel job | `kubectl delete pytorchjob <name> -n mlops-training` |

---

## Common Operations

### 1. Submitting a Training Job

#### Single GPU Training

```yaml
# training-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mistral-7b-lora-$(date +%Y%m%d-%H%M)
  namespace: mlops-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: llm-mlops/training:latest
            command:
            - python
            - -m
            - src.training.trainers.lora_trainer
            - --config=/configs/lora-7b.yaml
            - --data_path=s3://llm-mlops-data/training/
            - --output_dir=/checkpoints/
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: "32Gi"
            volumeMounts:
            - name: checkpoints
              mountPath: /checkpoints
            - name: config
              mountPath: /configs
          volumes:
          - name: checkpoints
            persistentVolumeClaim:
              claimName: training-checkpoints
          - name: config
            configMap:
              name: training-config
```

```bash
# Submit job
kubectl apply -f training-job.yaml

# Monitor status
kubectl get pytorchjob -n mlops-training -w
```

#### Multi-GPU Distributed Training

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llama-13b-distributed
  namespace: mlops-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: llm-mlops/training:latest
            env:
            - name: MASTER_ADDR
              value: "$(MASTER_ADDR)"
            resources:
              limits:
                nvidia.com/gpu: 4
    Worker:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: llm-mlops/training:latest
            resources:
              limits:
                nvidia.com/gpu: 4
```

---

### 2. Monitoring Training Progress

#### View Logs

```bash
# Master pod logs
kubectl logs -f mistral-7b-lora-master-0 -n mlops-training

# Worker pod logs (distributed training)
kubectl logs -f mistral-7b-lora-worker-0 -n mlops-training

# Stream all pod logs
kubectl logs -f -l training.kubeflow.org/job-name=mistral-7b-lora -n mlops-training
```

#### Check MLflow Metrics

```bash
# Port forward to MLflow
kubectl port-forward svc/mlflow -n mlflow 5000:5000

# Access at http://localhost:5000
# Navigate to experiment to see real-time metrics
```

#### GPU Monitoring

```bash
# Check GPU utilization
kubectl exec -it mistral-7b-lora-master-0 -n mlops-training -- nvidia-smi

# Watch GPU memory
kubectl exec -it mistral-7b-lora-master-0 -n mlops-training -- watch -n 1 nvidia-smi
```

---

### 3. Checkpoint Management

#### View Checkpoints

```bash
# List checkpoints in PVC
kubectl exec -it training-admin -n mlops-training -- ls -la /checkpoints/

# List checkpoints in S3
aws s3 ls s3://llm-mlops-artifacts/checkpoints/mistral-7b-lora/
```

#### Resume from Checkpoint

```yaml
# Add resume flag to job spec
command:
- python
- -m
- src.training.trainers.lora_trainer
- --config=/configs/lora-7b.yaml
- --resume_from_checkpoint=/checkpoints/checkpoint-500/
```

#### Backup Checkpoints to S3

```bash
# Manual backup
kubectl exec -it training-admin -n mlops-training -- \
  aws s3 sync /checkpoints/mistral-7b-lora/ s3://llm-mlops-artifacts/checkpoints/mistral-7b-lora/

# Automated backup (runs every 30 min)
# Configured in training-job.yaml with sidecar container
```

---

### 4. Canceling/Deleting Jobs

```bash
# Cancel running job
kubectl delete pytorchjob mistral-7b-lora -n mlops-training

# Force delete stuck job
kubectl delete pytorchjob mistral-7b-lora -n mlops-training --force --grace-period=0

# Clean up completed jobs (older than 1 day)
kubectl get pytorchjob -n mlops-training -o name | while read job; do
  age=$(kubectl get $job -n mlops-training -o jsonpath='{.status.completionTime}')
  # Delete if completed > 24h ago
done
```

---

## Incident Response

### Job Stuck in Pending

**Severity:** P2  
**Symptoms:** Job pods not starting

```bash
# 1. Check pod status
kubectl describe pod mistral-7b-lora-master-0 -n mlops-training

# 2. Common causes and solutions:
# - "Insufficient nvidia.com/gpu": Scale GPU node group
# - "node(s) had taint": Check node taints/tolerations
# - "PVC not bound": Check storage class and capacity

# 3. Check GPU availability
kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=\
  NAME:.metadata.name,\
  GPU:.status.allocatable.'nvidia\.com/gpu',\
  CONDITION:.status.conditions[-1].type

# 4. Scale GPU nodes if needed
eksctl scale nodegroup --cluster llm-mlops-dev --name gpu-training --nodes 4

# 5. Check Volcano scheduler (for gang scheduling)
kubectl logs -n volcano-system deployment/volcano-scheduler
```

### GPU Out of Memory (OOM)

**Severity:** P2  
**Alert:** Training pod OOMKilled or CUDA OOM

```bash
# 1. Check pod termination reason
kubectl describe pod mistral-7b-lora-master-0 -n mlops-training | grep -A5 "State:"

# 2. Check GPU memory before OOM
kubectl logs mistral-7b-lora-master-0 -n mlops-training --previous | grep -i "memory\|oom"

# 3. Solutions:
# Option A: Reduce batch size
# Option B: Enable gradient checkpointing
# Option C: Use DeepSpeed ZeRO-3
# Option D: Use QLoRA instead of LoRA

# 4. Update training config
kubectl edit configmap training-config -n mlops-training
# Change: batch_size: 4 -> batch_size: 2
# Add: gradient_checkpointing: true

# 5. Resubmit job
kubectl delete pytorchjob mistral-7b-lora -n mlops-training
kubectl apply -f training-job.yaml
```

### Training Loss Not Decreasing

**Severity:** P3  
**Symptoms:** Loss plateau, NaN loss

```bash
# 1. Check training logs for warnings
kubectl logs mistral-7b-lora-master-0 -n mlops-training | grep -i "warn\|nan\|inf"

# 2. Check MLflow for loss curve
# Port forward and view experiment

# 3. Common causes:
# - Learning rate too high -> Reduce by 10x
# - Data issues -> Check data preprocessing
# - Gradient explosion -> Enable gradient clipping

# 4. Recommended actions:
# - Review hyperparameters
# - Check data quality
# - Enable mixed precision (fp16) for stability
```

### Checkpoint Corruption

**Severity:** P2  
**Symptoms:** Resume fails, checkpoint loading errors

```bash
# 1. Verify checkpoint integrity
kubectl exec -it training-admin -n mlops-training -- \
  python -c "import torch; torch.load('/checkpoints/checkpoint-500/pytorch_model.bin')"

# 2. List available checkpoints
aws s3 ls s3://llm-mlops-artifacts/checkpoints/mistral-7b-lora/ --recursive

# 3. Find last valid checkpoint
for ckpt in checkpoint-500 checkpoint-400 checkpoint-300; do
  if aws s3 ls s3://llm-mlops-artifacts/checkpoints/mistral-7b-lora/$ckpt/; then
    echo "Valid checkpoint: $ckpt"
    break
  fi
done

# 4. Resume from earlier checkpoint
kubectl patch pytorchjob mistral-7b-lora -n mlops-training \
  --type='json' -p='[
    {"op": "replace", "path": "/spec/pytorchReplicaSpecs/Master/template/spec/containers/0/args/3", 
     "value": "--resume_from_checkpoint=/checkpoints/checkpoint-400/"}
  ]'
```

### Distributed Training NCCL Errors

**Severity:** P2  
**Symptoms:** NCCL timeout, communication errors

```bash
# 1. Check network connectivity between pods
kubectl exec mistral-7b-lora-master-0 -n mlops-training -- \
  ping mistral-7b-lora-worker-0.mistral-7b-lora.mlops-training.svc.cluster.local

# 2. Check NCCL environment
kubectl exec mistral-7b-lora-master-0 -n mlops-training -- env | grep NCCL

# 3. Common solutions:
# - Increase NCCL timeout: NCCL_TIMEOUT=3600
# - Use TCP instead of IB: NCCL_IB_DISABLE=1
# - Check security groups for inter-node communication

# 4. Add NCCL debugging
env:
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_TIMEOUT
  value: "3600"
```

---

## Performance Optimization

### DeepSpeed Configuration

```json
// configs/deepspeed-zero3.json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_num_steps": 100
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "overlap_comm": true
  },
  "fp16": {
    "enabled": true
  }
}
```

### Training Best Practices

| Aspect | Recommendation |
|--------|----------------|
| Batch Size | Start with 4, increase if GPU memory allows |
| Learning Rate | 1e-4 to 3e-4 for LoRA |
| Warmup | 5-10% of total steps |
| Gradient Clipping | 1.0 |
| Mixed Precision | FP16 for training, BF16 if available |
| Checkpointing | Save every 500 steps |

---

## Resource Management

### GPU Quotas

```yaml
# ResourceQuota for training namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: training-gpu-quota
  namespace: mlops-training
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
```

### Priority Classes

```yaml
# High priority for production training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-high-priority
value: 1000000
globalDefault: false
description: "High priority for critical training jobs"
```

---

## Monitoring

### Key Metrics

| Metric | Target | Alert |
|--------|--------|-------|
| Training Loss | Decreasing | Plateau > 1h |
| GPU Utilization | > 80% | < 50% sustained |
| Throughput | > 10 samples/sec | < 5 samples/sec |
| Checkpoint Frequency | Every 500 steps | Missing > 2h |

### MLflow Dashboards

- Training runs: `mlflow ui` at port 5000
- Metrics comparison across runs
- Model registry for completed models

---

## Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| ML Platform On-Call | PagerDuty | Immediate |
| Training Lead | Slack #ml-training | 15 min |
| Infrastructure | Slack #platform | 30 min |
