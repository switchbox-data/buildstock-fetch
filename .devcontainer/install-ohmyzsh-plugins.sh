#!/usr/bin/env bash
set -euo pipefail

# install-ohmyzsh-plugins.sh - Install custom oh-my-zsh plugins
#
# This script clones custom oh-my-zsh plugins for the current user.

echo "===================================================================="
echo "🔌 Installing custom oh-my-zsh plugins"
echo "===================================================================="
echo

PLUGINS_DIR="${HOME}/.oh-my-zsh/custom/plugins"

# Initialize counters for installed and existing plugins
INSTALLED_COUNT=0
EXISTING_COUNT=0

# Verify oh-my-zsh is installed
if [[ -d "${HOME}/.oh-my-zsh" ]]; then
  # Get oh-my-zsh version from git
  if [[ -d "${HOME}/.oh-my-zsh/.git" ]]; then
    # Try different methods to get version info
    if OMZ_VERSION=$(cd "${HOME}/.oh-my-zsh" && git describe --tags 2>/dev/null); then
      echo "✅ Found oh-my-zsh version: ${OMZ_VERSION}"
    elif OMZ_COMMIT=$(cd "${HOME}/.oh-my-zsh" && git rev-parse --short HEAD 2>/dev/null); then
      echo "✅ Found oh-my-zsh commit: ${OMZ_COMMIT}"
    else
      echo "✅ Found oh-my-zsh at ${HOME}/.oh-my-zsh"
    fi
  else
    echo "✅ Found oh-my-zsh at ${HOME}/.oh-my-zsh"
  fi
else
  echo "❌ ERROR: oh-my-zsh not found at ${HOME}/.oh-my-zsh" >&2
  echo "The zsh feature must be installed first" >&2
  exit 1
fi
echo

# Create plugins directory if it doesn't exist
if [[ -d "${PLUGINS_DIR}" ]]; then
  echo "✅ Found plugins directory: ${PLUGINS_DIR}"
else
  echo "📁 Creating plugins directory: ${PLUGINS_DIR}"
  mkdir -p "${PLUGINS_DIR}"
fi
echo

# Clone zsh-autosuggestions
if [[ -d "${PLUGINS_DIR}/zsh-autosuggestions" ]]; then
  echo "✅ zsh-autosuggestions already installed"
  EXISTING_COUNT=$((EXISTING_COUNT + 1))
else
  echo "📦 Cloning zsh-autosuggestions..."
  git clone --quiet https://github.com/zsh-users/zsh-autosuggestions.git \
    "${PLUGINS_DIR}/zsh-autosuggestions"
  INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Clone fast-syntax-highlighting
if [[ -d "${PLUGINS_DIR}/fast-syntax-highlighting" ]]; then
  echo "✅ fast-syntax-highlighting already installed"
  EXISTING_COUNT=$((EXISTING_COUNT + 1))
else
  echo "📦 Cloning fast-syntax-highlighting..."
  git clone --quiet https://github.com/zdharma-continuum/fast-syntax-highlighting.git \
    "${PLUGINS_DIR}/fast-syntax-highlighting"
  INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

# Clone zsh-autocomplete
if [[ -d "${PLUGINS_DIR}/zsh-autocomplete" ]]; then
  echo "✅ zsh-autocomplete already installed"
  EXISTING_COUNT=$((EXISTING_COUNT + 1))
else
  echo "📦 Cloning zsh-autocomplete..."
  git clone --quiet --depth 1 https://github.com/marlonrichert/zsh-autocomplete.git \
    "${PLUGINS_DIR}/zsh-autocomplete"
  INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
fi

echo

# Display summary
if [[ $INSTALLED_COUNT -gt 0 ]] && [[ $EXISTING_COUNT -gt 0 ]]; then
  echo "✅ Installed ${INSTALLED_COUNT} plugin(s), ${EXISTING_COUNT} already installed"
elif [[ $INSTALLED_COUNT -gt 0 ]]; then
  echo "✅ Installed ${INSTALLED_COUNT} plugin(s)"
else
  echo "✅ ${EXISTING_COUNT} plugin(s) already installed"
fi
echo
echo "===================================================================="
echo "✨ oh-my-zsh plugins installed successfully!"
echo "===================================================================="
echo
echo
