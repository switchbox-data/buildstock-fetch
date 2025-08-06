#! /usr/bin/env bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Dependencies
uv sync

# Install the package in development mode so the CLI command is available
uv pip install -e .

# Add the virtual environment to PATH so CLI commands are available
echo 'export PATH="/workspaces/buildstock-fetch/.venv/bin:$PATH"' >> ~/.bashrc

# Install pre-commit hooks
git config --global --add safe.directory /workspaces/buildstock-fetch # Needed for pre-commit install to work
uv run pre-commit install --install-hooks
