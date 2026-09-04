#!/bin/bash
# Build a double-clickable macOS app bundle around the agent binary.
# Usage: macos-app.sh <binary> <version> <output.app>
set -euo pipefail

binary="${1:?binary}"
version="${2:?version}"
app="${3:?output.app}"

mkdir -p "${app}/Contents/MacOS"
cp "${binary}" "${app}/Contents/MacOS/rungpu-agent"
chmod +x "${app}/Contents/MacOS/rungpu-agent"

cat > "${app}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>RunGPU Agent</string>
  <key>CFBundleDisplayName</key>
  <string>RunGPU Agent</string>
  <key>CFBundleIdentifier</key>
  <string>io.rungpu.agent</string>
  <key>CFBundleVersion</key>
  <string>${version}</string>
  <key>CFBundleShortVersionString</key>
  <string>${version}</string>
  <key>CFBundleExecutable</key>
  <string>rungpu-agent</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF
