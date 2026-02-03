#!/usr/bin/env bash
set -euo pipefail

# install-prek.sh - Install prek CLI for pre-commit hooks
#
# This script installs prek for the current user.
# The actual hook installation (prek install --install-hooks) happens
# after the container starts, when .git/ is mounted from the host.

echo "===================================================================="
echo "🪝 Installing prek pre-commit hook framework"
echo "===================================================================="
echo

# Install prek
echo "📦 Installing prek"
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/j178/prek/releases/download/v0.2.11/prek-installer.sh | sh
echo

# Add prek to PATH for this session
# The installer adds prek to ~/.local/bin, so we add that to PATH
export PATH="${HOME}/.local/bin:${PATH}"

# Verify installation
if ! PREK_VERSION=$(prek --version 2>&1); then
  echo "❌ ERROR: prek installation failed or prek command not found" >&2
  echo "Expected location: ${HOME}/.local/bin/prek" >&2
  exit 1
fi

echo "✅ Installed: ${PREK_VERSION}"

echo
echo "💡 To install pre-commit hooks:"
echo "   prek install --install-hooks"
echo
echo "===================================================================="
echo "✨ prek installed successfully!"
echo "===================================================================="
echo
echo
