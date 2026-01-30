#! /usr/bin/env bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Dependencies
uv sync

# Install the package in development mode so the CLI command is available
uv pip install -e .

# Install lint dependencies (including pre-commit)
uv pip install -e ".[lint]"

# Add the virtual environment to PATH so CLI commands are available
echo 'export PATH="/workspaces/buildstock-fetch/.venv/bin:$PATH"' >>~/.bashrc

# Install prek
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/download/v0.2.11/prek-installer.sh | sh

# Install pre-commit hooks
prek install --install-hooks
