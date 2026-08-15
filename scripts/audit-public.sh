#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

report_matches() {
    local description="$1"
    local pattern="$2"
    shift 2
    local matches
    matches=$(git grep --no-index -nEI "$pattern" -- "$@" 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        echo "ERROR: $description"
        echo "$matches"
        fail=1
    fi
}

forbidden_files=$(
    git ls-files --cached --others --exclude-standard | grep -E '(^|/)(\.env($|\.)|coverage\.out$|config\.ya?ml$|.*\.(pem|key|p12|pfx)$|outbox/|dist/)' |
    while IFS= read -r file; do
        [[ -e "$file" ]] && echo "$file"
    done
    true
)
if [[ -n "$forbidden_files" ]]; then
    echo "ERROR: generated state or credential-bearing files are present:"
    echo "$forbidden_files"
    fail=1
fi

report_matches "a live credential-shaped value is present" \
    '(sk_live_[A-Za-z0-9]{16,}|sk_test_[A-Za-z0-9]{16,}|pk_live_[A-Za-z0-9]{16,}|sbp_[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|postgres(ql)?://[^[:space:]/]+:[^[:space:]@]+@)' \
    .

report_matches "coordinator-only policy leaked into public agent source" \
    '(PLATFORM_FEE_PERCENT|TARGET_MARGIN|gpu_owner_share|settlement|stripe_transfer|payout_status|load-balancer|queue-dispatcher)' \
    '*.go' '*.sh' '*.ps1' ':(exclude)scripts/audit-public.sh'

if git grep --no-index -nE 'json:"(cpu_model|cpu_cores|os_info|ram_used_gb|cpu_percent|hostname)' -- '*.go' >/tmp/rungpu-public-audit.$$ 2>/dev/null; then
    echo "ERROR: unnecessary host fingerprint telemetry is present:"
    cat /tmp/rungpu-public-audit.$$
    fail=1
fi
rm -f /tmp/rungpu-public-audit.$$

if [[ $fail -ne 0 ]]; then
    exit 1
fi

echo "Public agent audit passed."
