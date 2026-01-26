#!/bin/bash
# Install project dependencies
# Usage: ./install-dependencies.sh

set -euo pipefail

echo "==> Installing Python dependencies..."

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Created virtual environment"
fi

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
echo "==> Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

echo ""
echo "==> Dependencies installed!"
echo ""
echo "To activate the virtual environment:"
echo "    source venv/bin/activate"
