#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker run --rm --pull always \
    --volume "$PWD:/src:ro" \
    golang:1.22-bookworm \
    bash -euo pipefail -c '
        cp -a /src /work
        cd /work
        test ! -e "$(command -v ollama 2>/dev/null || true)"
        ./scripts/install-ollama-linux.sh
        useradd --create-home --uid 1001 agent
        chown -R agent:agent /work
        runuser -u agent -- env HOME=/home/agent PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/go/bin \
            RUNGPU_RUNTIME_SMOKE=1 go test -v -count=1 \
            -run "^TestRuntimeSmoke_NativeOllamaInference$" ./internal/job -timeout 12m
    '