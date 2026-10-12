#!/bin/bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
        echo "Usage: bash scripts/test-docker-inference.sh"
        echo "Requires this source checkout, Go 1.22+, and a running Linux-container Docker engine."
        echo "Runs real CPU inference in Docker; no GPU, Ollama, enrollment, or RunGPU key needed."
        exit 0
    fi
    echo "Error: unexpected arguments. Use --help for usage." >&2
    exit 2
fi

cd "$(dirname "$0")/.."

for tool in go docker tee grep mktemp; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: required command '$tool' is missing. Install prerequisites and retry." >&2
        exit 1
    fi
done

if ! docker version; then
    echo "Error: Docker is not reachable. Start Docker Desktop or Docker Engine and retry." >&2
    echo "Use Linux containers, and ensure this user can access the Docker daemon." >&2
    exit 1
fi
go version

log_file="$(mktemp "${TMPDIR:-/tmp}/rungpu-docker-inference.XXXXXX")"
echo "Running real CPU Docker inference with a small public model."
echo "This may download Go dependencies, the pinned container image, and a roughly 100 MB model."
echo "No enrollment, customer jobs, billing, or runtime installation is performed."
echo "Log: $log_file"

# Ignore inherited test flags that could skip the real check or reuse cached results.
if ! GOFLAGS='' RUNGPU_RUNTIME_SMOKE=1 go test -v -count=1 -short=false \
    -run '^TestRuntimeSmoke_DockerInference$' ./internal/job -timeout 20m \
    2>&1 | tee "$log_file"; then
    echo "FAIL: Docker inference check failed. See $log_file" >&2
    exit 1
fi

if ! grep -Eq '^--- PASS: TestRuntimeSmoke_DockerInference \(' "$log_file"; then
    echo "FAIL: the real Docker test did not report a pass (it may have been skipped). See $log_file" >&2
    exit 1
fi

echo "PASS: a real model ran in Docker and produced the expected answer. Log: $log_file"
