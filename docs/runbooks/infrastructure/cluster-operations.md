# Cluster Operations Runbook

> **Last Updated:** 2026-01-29  
> **Owner:** Platform Team  
> **Severity Levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

## Overview

This runbook covers operational procedures for the EKS cluster including scaling, upgrades, node management, and disaster recovery.

---

## Quick Reference

| Action | Command |
|--------|---------|
| Get cluster status | `kubectl get nodes` |
| Check node health | `kubectl describe node <node>` |
| View cluster events | `kubectl get events --sort-by='.lastTimestamp'` |
| Check GPU nodes | `kubectl get nodes -l nvidia.com/gpu.present=true` |

---

## Common Operations

### 1. Checking Cluster Health

**When to use:** Regular health checks, incident investigation

```bash
# Overall cluster status
kubectl cluster-info

# Node status
kubectl get nodes -o wide

# Resource utilization
kubectl top nodes
kubectl top pods --all-namespaces

# Pending pods (possible resource constraints)
kubectl get pods --all-namespaces --field-selector=status.phase=Pending
```

**Expected output:** All nodes `Ready`, no pending pods due to resource constraints.

---

### 2. Node Scaling

#### Scale Up Node Group

**When to use:** Increased workload, training jobs queued

```bash
# Using eksctl
eksctl scale nodegroup \
  --cluster llm-mlops-dev \
  --name gpu-workers \
  --nodes 4 \
  --nodes-min 2 \
  --nodes-max 8

# Using AWS CLI
aws eks update-nodegroup-config \
  --cluster-name llm-mlops-dev \
  --nodegroup-name gpu-workers \
  --scaling-config minSize=2,maxSize=8,desiredSize=4
```

#### Scale Down Node Group

**When to use:** Cost optimization, reduced workload

> [!WARNING]
> Ensure no critical workloads are running on nodes being removed.

```bash
# Cordon nodes first (prevent new pods)
kubectl cordon <node-name>

# Drain workloads (graceful eviction)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Then scale down
eksctl scale nodegroup --cluster llm-mlops-dev --name gpu-workers --nodes 2
```

---

### 3. Node Replacement

**When to use:** Unhealthy node, instance type change

```bash
# Step 1: Cordon the problematic node
kubectl cordon <node-name>

# Step 2: Drain workloads
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force

# Step 3: Terminate instance (AWS will replace with healthy node)
INSTANCE_ID=$(kubectl get node <node-name> -o jsonpath='{.spec.providerID}' | cut -d'/' -f5)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Step 4: Verify new node joins
kubectl get nodes -w
```

---

### 4. Cluster Upgrades

#### Pre-Upgrade Checklist

- [ ] Review [EKS release notes](https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html)
- [ ] Check addon compatibility
- [ ] Backup critical resources (Velero)
- [ ] Notify stakeholders
- [ ] Schedule maintenance window

#### Upgrade Control Plane

```bash
# Check current version
kubectl version --short

# Upgrade control plane (one minor version at a time)
aws eks update-cluster-version \
  --name llm-mlops-dev \
  --kubernetes-version 1.30

# Monitor upgrade (takes 20-40 minutes)
aws eks describe-update \
  --name llm-mlops-dev \
  --update-id <update-id>
```

#### Upgrade Node Groups

```bash
# Update launch template with new AMI
# Then trigger rolling update
eksctl upgrade nodegroup \
  --cluster llm-mlops-dev \
  --name system-workers \
  --kubernetes-version 1.30
```

#### Upgrade Add-ons

```bash
# List current add-on versions
aws eks list-addons --cluster-name llm-mlops-dev

# Upgrade VPC CNI
aws eks update-addon \
  --cluster-name llm-mlops-dev \
  --addon-name vpc-cni \
  --addon-version v1.16.0-eksbuild.1

# Upgrade CoreDNS
aws eks update-addon \
  --cluster-name llm-mlops-dev \
  --addon-name coredns \
  --addon-version v1.11.1-eksbuild.4
```

---

## Incident Response

### Node Not Ready

**Severity:** P1  
**Symptoms:** Node shows `NotReady` status

```bash
# 1. Check node status
kubectl describe node <node-name>

# 2. Check kubelet logs (if SSH access available)
journalctl -u kubelet -f

# 3. Check system pods on node
kubectl get pods --all-namespaces -o wide --field-selector spec.nodeName=<node-name>

# 4. If disk pressure:
kubectl get node <node-name> -o jsonpath='{.status.conditions[?(@.type=="DiskPressure")].status}'

# 5. Resolution options:
# Option A: Wait for self-healing (5 min)
# Option B: Drain and terminate (see Node Replacement)
# Option C: SSH and restart kubelet
```

### GPU Driver Issues

**Severity:** P1  
**Symptoms:** GPU pods stuck in Pending, nvidia-smi errors

```bash
# 1. Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# 2. Check GPU resource on node
kubectl describe node <gpu-node> | grep nvidia.com/gpu

# 3. Check device plugin logs
kubectl logs -n kube-system <nvidia-device-plugin-pod>

# 4. Restart device plugin
kubectl delete pod -n kube-system -l name=nvidia-device-plugin-ds

# 5. If persists, reboot node
kubectl drain <node> --ignore-daemonsets
aws ec2 reboot-instances --instance-ids <instance-id>
```

### API Server Unresponsive

**Severity:** P0  
**Symptoms:** kubectl timeout, 5xx errors

```bash
# 1. Check from AWS Console: EKS > Cluster > Overview

# 2. Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/eks/llm-mlops

# 3. Check API server metrics
# Access via CloudWatch Insights

# 4. Common causes:
# - Control plane upgrade in progress
# - IAM authentication issues
# - Network connectivity (VPC endpoints)

# 5. Contact AWS Support if P0 and persists > 10 min
```

---

## Disaster Recovery

### Backup Procedures

```bash
# Full cluster backup with Velero
velero backup create full-backup-$(date +%Y%m%d) \
  --include-namespaces mlops-system,mlops-training,mlops-serving

# Backup specific namespace
velero backup create mlflow-backup \
  --include-namespaces mlflow

# List backups
velero backup get
```

### Restore Procedures

```bash
# Restore from backup
velero restore create --from-backup full-backup-20260129

# Restore specific namespace
velero restore create --from-backup full-backup-20260129 \
  --include-namespaces mlflow

# Monitor restore
velero restore describe <restore-name>
```

### Cluster Recreation

In case of complete cluster loss:

```bash
# 1. Recreate infrastructure
cd terraform/environments/dev
terraform apply

# 2. Bootstrap ArgoCD
kubectl apply -k kubernetes/argocd/install/

# 3. Sync applications
argocd app sync app-of-apps --prune

# 4. Restore data
velero restore create --from-backup latest-backup
```

---

## Maintenance Windows

### Scheduled Maintenance

| Day | Time (UTC) | Activity |
|-----|------------|----------|
| Sunday | 02:00-06:00 | Cluster upgrades |
| Daily | 03:00 | Automated backups |

### Pre-Maintenance Checklist

- [ ] Notify #platform-announcements Slack channel
- [ ] Scale down non-critical workloads
- [ ] Verify backup completion
- [ ] Review pending alerts

### Post-Maintenance Checklist

- [ ] Verify all nodes Ready
- [ ] Check critical services health
- [ ] Run E2E smoke tests
- [ ] Update incident channel with status

---

## Contacts

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-Call Engineer | PagerDuty | Immediate |
| Platform Lead | @platform-lead | 15 min |
| AWS Support | Support Console | P0: Immediate |
