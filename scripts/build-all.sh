#!/bin/bash
#
# Cross-compiles the RunGPU agent for Linux, Windows, and macOS.
# Output: dist/rungpu-agent-<os>-<arch>[.exe]
#
# Go produces static binaries — no runtime dependencies on the target host.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p dist

VERSION="${VERSION:-dev}"
BIN="rungpu-agent"
LDFLAGS="-s -w -X main.version=${VERSION}"

build() {
    local goos="$1" goarch="$2" ext="${3:-}"
    local out="dist/${BIN}-${goos}-${goarch}${ext}"
    echo "  Building ${goos}/${goarch}..."
    CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" \
        go build -ldflags "$LDFLAGS" -o "$out" .
    echo "    → $out ($(du -h "$out" | cut -f1))"
}

echo "Building RunGPU Agent v${VERSION}"
echo ""

build linux   amd64
build linux   arm64
build windows amd64 .exe
build darwin  amd64
build darwin  arm64

echo ""
echo "Done. Binaries:"
ls -lh dist/
echo ""
echo "To create a release archive:"
echo "  cd dist && for f in rungpu-agent-*; do tar czf \"\${f%.exe}.tar.gz\" \"\$f\"; done"
