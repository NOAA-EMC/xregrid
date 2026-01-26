#!/bin/bash
set -e

echo "XRegrid Documentation Builder"
echo "=============================="

# Check if we are in a conda environment, if not try to activate
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Activating conda environment..."
    # Attempt to activate, but don't fail if conda is not installed
    command -v conda >/dev/null 2>&1 && conda activate xregrid || echo "Warning: Could not activate conda environment. Continuing with current environment."
fi

# Install documentation dependencies if needed
echo "Checking documentation dependencies..."
pip install mkdocs mkdocs-material mkdocs-gallery mkdocs-autorefs mkdocstrings mkdocstrings-python matplotlib cartopy pooch --quiet

# Install package in development mode
echo "Installing xregrid package..."
# We use --no-deps because we assume environment is already set up or we don't want to fail on binary deps like esmpy
pip install -e . --quiet --no-deps || echo "Warning: Could not install in editable mode."

# Clean previous build
echo "Cleaning previous build..."
rm -rf site/

# Build documentation
echo "Building documentation..."
# Workaround for Python 3.14 compatibility with mkdocs-gallery
# ast.Str was removed in 3.14, so we monkeypatch it if necessary
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -c "import ast; \
ast.Str = getattr(ast, 'Str', ast.Constant); \
ast.Num = getattr(ast, 'Num', ast.Constant); \
ast.Bytes = getattr(ast, 'Bytes', ast.Constant); \
ast.NameConstant = getattr(ast, 'NameConstant', ast.Constant); \
from mkdocs.__main__ import cli; cli()" build --clean

echo ""
echo "Documentation built successfully!"
echo "To serve locally: mkdocs serve"
echo "To deploy: mkdocs gh-deploy"
echo ""
echo "Built site is in: ./site/"
