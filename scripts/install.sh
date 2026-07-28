#!/bin/bash
#
# Installs the Tokenize GPU Agent (Go) on Linux as a systemd service.
# Usage: sudo ./scripts/install.sh [API_KEY]
set -euo pipefail

BIN_NAME="tokenize-gpu-agent"
INSTALL_DIR="/usr/local/bin"
SERVICE="/etc/systemd/system/tokenize-gpu-agent.service"
RUN_USER="${SUDO_USER:-$USER}"
API_KEY="${1:-}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run with sudo."
    exit 1
fi

if ! command -v go >/dev/null 2>&1; then
    echo "ERROR: Go toolchain not found. Install Go 1.22+ or drop a prebuilt binary into $INSTALL_DIR."
    exit 1
fi

command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found; agent will run CPU-only."
command -v docker     >/dev/null 2>&1 || echo "WARNING: docker not found; agent cannot run container jobs."

cd "$(dirname "$0")/.."
echo "Building agent..."
sudo -u "$RUN_USER" env "PATH=$PATH" go build -ldflags "-s -w" -o "$BIN_NAME" .

echo "Installing to $INSTALL_DIR..."
install -m 0755 "$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"

if [[ -n "$API_KEY" ]]; then
    sudo -u "$RUN_USER" "$INSTALL_DIR/$BIN_NAME" init --api-key "$API_KEY"
fi

echo "Writing systemd unit..."
cat > "$SERVICE" <<UNIT
[Unit]
Description=Tokenize GPU Pool Agent
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
ExecStart=${INSTALL_DIR}/${BIN_NAME} start
Restart=always
RestartSec=5
SupplementaryGroups=docker
Environment=GODEBUG=

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable tokenize-gpu-agent
systemctl restart tokenize-gpu-agent

echo ""
echo "Installed. Logs:   journalctl -u tokenize-gpu-agent -f"
echo "          Status:  systemctl status tokenize-gpu-agent"
[[ -z "$API_KEY" ]] && echo "Next: $BIN_NAME init --api-key YOUR_API_KEY && systemctl restart tokenize-gpu-agent"
