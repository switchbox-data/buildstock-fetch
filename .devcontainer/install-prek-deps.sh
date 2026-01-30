#!/usr/bin/env bash
set -euo pipefail

# install-prek-deps.sh - Install prek pre-commit hooks
#
# Requirements:
#   - prek must be installed
#   - .pre-commit-config.yaml file must exist

echo "===================================================================="
echo "🪝 Installing prek pre-commit hooks from .pre-commit-config.yaml"
echo "===================================================================="
echo

# Find the git repository root and cd there
# This works in both devcontainers and on laptops
REPO_ROOT=$(git rev-parse --show-toplevel 2>&1) || {
  # Git may fail with "dubious ownership" in containers where mounted files
  # are owned by a different user than the running process. This happens because:
  #   - Docker Desktop on Mac maps host user UIDs to container UIDs (often UID 1000)
  #   - The container runs as root (UID 0)
  #   - Git sees the mismatch and refuses to operate (security feature from CVE-2022-24765)
  # The safe.directory config tells git to trust this specific directory.
  # We only add it when the dubious ownership error occurs, not unconditionally.
  if echo "$REPO_ROOT" | grep -q "dubious ownership"; then
    git config --global --add safe.directory "$(pwd)"
    REPO_ROOT=$(git rev-parse --show-toplevel)
  else
    echo "❌ ERROR: Not in a git repository" >&2
    exit 1
  fi
}
cd "${REPO_ROOT}"
echo "📁 Working in repository: ${REPO_ROOT}"
echo

# Check if .pre-commit-config.yaml exists
PRECOMMIT_CONFIG=".pre-commit-config.yaml"
if [ -f "${PRECOMMIT_CONFIG}" ]; then
  echo "✅ Found .pre-commit-config.yaml: $(realpath "${PRECOMMIT_CONFIG}")"
else
  echo "❌ ERROR: .pre-commit-config.yaml is necessary but not found in $(pwd)" >&2
  exit 1
fi
echo

# Check if prek is installed
echo "📦 Checking for prek..."
echo
if command -v prek >/dev/null 2>&1; then
  PREK_VERSION=$(prek --version 2>&1)
  echo "✅ Using: ${PREK_VERSION}"
else
  echo "❌ ERROR: prek is not installed" >&2
  echo "Expected location: ${HOME}/.local/bin/prek" >&2
  echo "" >&2
  echo "Install prek first by running: .devcontainer/install-prek.sh" >&2
  exit 1
fi
echo

# Check if pre-commit hooks are already installed
PRECOMMIT_HOOK=".git/hooks/pre-commit"
HOOKS_ALREADY_INSTALLED=false

if [ -f "${PRECOMMIT_HOOK}" ]; then
  # Check if it's a prek-managed hook
  if grep -q "prek" "${PRECOMMIT_HOOK}" 2>/dev/null; then
    HOOKS_ALREADY_INSTALLED=true
  fi
fi

# Count total hooks from .pre-commit-config.yaml (look for "- id:" pattern)
TOTAL_HOOKS=$(grep -c "^\s*- id:" .pre-commit-config.yaml 2>/dev/null || echo "0")
# Ensure it's a single integer
TOTAL_HOOKS=$(echo "$TOTAL_HOOKS" | head -1 | tr -d ' ')

# Install prek pre-commit hooks
if [ "$HOOKS_ALREADY_INSTALLED" = true ]; then
  if [ "$TOTAL_HOOKS" -gt 0 ] 2>/dev/null; then
    echo "✅ ${TOTAL_HOOKS} pre-commit hooks already installed"
  else
    echo "✅ Pre-commit hooks already installed"
  fi
else
  echo "📥 Installing prek pre-commit hooks..."
  PREK_OUTPUT=$(prek install --install-hooks 2>&1)
  echo "$PREK_OUTPUT"
  echo

  if [ "$TOTAL_HOOKS" -gt 0 ] 2>/dev/null; then
    echo "✅ Installed ${TOTAL_HOOKS} pre-commit hooks"
  else
    echo "✅ Pre-commit hooks installed"
  fi
fi
echo

echo "===================================================================="
echo "✨ prek pre-commit hooks installed successfully!"
echo "===================================================================="
echo
echo
