.PHONY: install lint test format terraform-init terraform-plan terraform-apply deploy-dev clean help

# Python environment
PYTHON := python
PIP := pip
VENV := venv

# Terraform
TF := terraform
TF_ENV ?= dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Development Setup
# ============================================================================

install: ## Install all dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	pre-commit install

venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Run 'source $(VENV)/bin/activate' to activate"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linting checks
	ruff check src/ tests/ pipelines/
	mypy src/ --ignore-missing-imports

format: ## Format code
	ruff format src/ tests/ pipelines/
	ruff check --fix src/ tests/ pipelines/

test: ## Run tests
	pytest tests/unit -v --cov=src --cov-report=term-missing

test-integration: ## Run integration tests
	pytest tests/integration -v

test-all: ## Run all tests
	pytest tests/ -v --cov=src

# ============================================================================
# Terraform
# ============================================================================

terraform-init: ## Initialize Terraform for environment (TF_ENV=dev|staging|prod)
	cd terraform/environments/$(TF_ENV) && $(TF) init

terraform-plan: ## Plan Terraform changes
	cd terraform/environments/$(TF_ENV) && $(TF) plan -out=tfplan

terraform-apply: ## Apply Terraform changes
	cd terraform/environments/$(TF_ENV) && $(TF) apply tfplan

terraform-destroy: ## Destroy Terraform infrastructure
	cd terraform/environments/$(TF_ENV) && $(TF) destroy

terraform-validate: ## Validate all Terraform modules
	@for dir in terraform/modules/*; do \
		echo "Validating $$dir..."; \
		cd $$dir && $(TF) init -backend=false && $(TF) validate && cd -; \
	done

terraform-fmt: ## Format Terraform files
	$(TF) fmt -recursive terraform/

# ============================================================================
# Kubernetes
# ============================================================================

k8s-apply-base: ## Apply base Kubernetes resources
	kubectl apply -k kubernetes/base/

k8s-apply-dev: ## Apply dev overlay
	kubectl apply -k kubernetes/overlays/dev/

port-forward-grafana: ## Port forward Grafana
	kubectl port-forward svc/grafana -n monitoring 3000:80

port-forward-mlflow: ## Port forward MLflow
	kubectl port-forward svc/mlflow -n mlflow 5000:5000

port-forward-argocd: ## Port forward ArgoCD
	kubectl port-forward svc/argocd-server -n argocd 8080:443

# ============================================================================
# Docker
# ============================================================================

docker-build-training: ## Build training image
	docker build -t llm-mlops/training:latest -f docker/training/Dockerfile .

docker-build-serving: ## Build serving image
	docker build -t llm-mlops/serving:latest -f docker/serving/Dockerfile .

docker-build-all: docker-build-training docker-build-serving ## Build all images

# ============================================================================
# Deployment
# ============================================================================

deploy-dev: terraform-init terraform-plan terraform-apply ## Deploy dev environment
	@echo "Dev environment deployed"

deploy-argocd: ## Deploy ArgoCD
	./scripts/deploy/deploy-argocd.sh

bootstrap-cluster: ## Bootstrap cluster with ArgoCD apps
	./scripts/deploy/bootstrap-cluster.sh

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
