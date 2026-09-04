#!/bin/bash
set -euo pipefail

if command -v ollama >/dev/null 2>&1; then
    exit 0
fi

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "ERROR: Ollama installation requires root. Run with sudo." >&2
    exit 1
fi

missing=()
command -v curl >/dev/null 2>&1 || missing+=(curl)
command -v zstd >/dev/null 2>&1 || missing+=(zstd)

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Installing Ollama prerequisites: ${missing[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates "${missing[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y ca-certificates "${missing[@]}"
    elif command -v yum >/dev/null 2>&1; then
        yum install -y ca-certificates "${missing[@]}"
    else
        echo "ERROR: install curl and zstd, then retry." >&2
        exit 1
    fi
fi

installer=$(mktemp)
trap 'rm -f "$installer"' EXIT
curl --proto '=https' --tlsv1.2 -fsSL https://ollama.com/install.sh -o "$installer"
sh "$installer"

command -v ollama >/dev/null 2>&1 || {
    echo "ERROR: Ollama installer completed but ollama is not in PATH." >&2
    exit 1
}

ollama --version