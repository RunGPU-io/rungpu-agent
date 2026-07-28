#!/bin/bash
#
# Installs the Tokenize GPU Agent (Go) on macOS as a launchd LaunchAgent.
# Usage: ./scripts/install-macos.sh [API_KEY]
#
# On Apple Silicon Macs, the agent runs inference via Ollama (Metal backend).
# On Intel Macs, the agent runs inference via Ollama (CPU backend).
set -euo pipefail

BIN_NAME="tokenize-gpu-agent"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
LABEL="com.tokenize.gpu-agent"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
API_KEY="${1:-}"

# ── Platform checks ──────────────────────────────────────────────────────────

[[ "$(uname)" == "Darwin" ]] || { echo "ERROR: macOS only. Use install.sh on Linux."; exit 1; }
command -v go >/dev/null 2>&1 || { echo "ERROR: Go 1.22+ required (https://go.dev/dl)."; exit 1; }

# Detect Apple Silicon vs Intel
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Apple Silicon detected — Metal GPU acceleration available."
else
    echo "ℹ️  Intel Mac detected — CPU-only inference (no Metal)."
fi

# ── Ollama check ─────────────────────────────────────────────────────────────

if command -v ollama >/dev/null 2>&1; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo "✅ Ollama found: $OLLAMA_VERSION"
else
    echo ""
    echo "⚠️  Ollama is not installed."
    echo "   The agent uses Ollama to run inference jobs on macOS."
    echo ""
    echo "   Install it now:"
    echo "     brew install ollama"
    echo "   or download from: https://ollama.com/download/mac"
    echo ""
    read -rp "Continue without Ollama? (y/N) " CONTINUE
    if [[ "${CONTINUE,,}" != "y" ]]; then
        echo "Aborted. Install Ollama first, then re-run this script."
        exit 1
    fi
fi

# Check if Ollama server is running
if command -v ollama >/dev/null 2>&1; then
    if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "✅ Ollama server is running."
    else
        echo "⚠️  Ollama is installed but the server is not running."
        echo "   Start it with: ollama serve"
        echo "   Or it will start automatically when the agent sends a request."
    fi
fi

# Docker is optional on macOS (used only for CUDA, which macOS doesn't support)
if command -v docker >/dev/null 2>&1; then
    echo "ℹ️  Docker found (not required on macOS — jobs run via Ollama)."
else
    echo "ℹ️  Docker not found (not required on macOS — jobs run via Ollama)."
fi

# ── Build ────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."
echo ""
echo "Building agent for macOS/$ARCH..."
CGO_ENABLED=0 go build -ldflags "-s -w" -o "$BIN_NAME" .

echo "Installing to $INSTALL_DIR (may require sudo)..."
sudo mkdir -p "$INSTALL_DIR"
sudo install -m 0755 "$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"

# ── Init config ──────────────────────────────────────────────────────────────

if [[ -n "$API_KEY" ]]; then
    echo "Initializing config..."
    "$INSTALL_DIR/$BIN_NAME" init --api-key "$API_KEY"
fi

# ── Show detected hardware ───────────────────────────────────────────────────

echo ""
echo "Detected hardware:"
"$INSTALL_DIR/$BIN_NAME" status 2>/dev/null || true

# ── Create LaunchAgent ───────────────────────────────────────────────────────

echo ""
echo "Creating LaunchAgent at $PLIST..."
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/${BIN_NAME}</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${HOME}/Library/Logs/tokenize-gpu-agent.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/Library/Logs/tokenize-gpu-agent.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Tokenize GPU Agent installed on macOS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Binary:  $INSTALL_DIR/$BIN_NAME"
echo "  Config:  ~/.tokenize/config.yaml"
echo "  Logs:    ~/Library/Logs/tokenize-gpu-agent.log"
echo "  Backend: Ollama (Metal on Apple Silicon, CPU on Intel)"
echo ""
echo "  Stop:    launchctl unload $PLIST"
echo "  Start:   launchctl load $PLIST"
echo "  Status:  $BIN_NAME status"
echo ""
if [[ -z "$API_KEY" ]]; then
    echo "  ⚠️  Next: $BIN_NAME init --api-key YOUR_API_KEY"
    echo ""
fi
echo "  Ensure Ollama is running: ollama serve"
echo ""
