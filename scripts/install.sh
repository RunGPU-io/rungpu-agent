#!/bin/bash
#
# Installs the Tokenize GPU Agent (Go) on Linux as a systemd service.
# Enroll after installation with a one-time token from the host dashboard.
set -euo pipefail

BIN_NAME="tokenize-gpu-agent"
INSTALL_DIR="/usr/local/bin"
SERVICE="/etc/systemd/system/tokenize-gpu-agent.service"
RUN_USER="${SUDO_USER:-$USER}"

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

if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama for text inference..."
    "$(dirname "$0")/install-ollama-linux.sh"
else
    echo "Ollama already installed: $(ollama --version 2>/dev/null || echo unknown version)"
fi

cd "$(dirname "$0")/.."
echo "Building agent..."
sudo -u "$RUN_USER" env "PATH=$PATH" go build -ldflags "-s -w" -o "$BIN_NAME" .

echo "Installing to $INSTALL_DIR..."
install -m 0755 "$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"

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
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable tokenize-gpu-agent

echo ""
echo "Installed. Logs:   journalctl -u tokenize-gpu-agent -f"
echo "          Status:  systemctl status tokenize-gpu-agent"
echo "Next: sudo -u $RUN_USER $BIN_NAME init --enrollment-token YOUR_ONE_TIME_TOKEN"
echo "Then: systemctl start tokenize-gpu-agent"
