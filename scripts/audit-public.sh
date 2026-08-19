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

report_matches_except_literal() {
    local description="$1"
    local pattern="$2"
    local allowed_literal="$3"
    shift 3
    local matches
    matches=$(git grep --no-index -nEI "$pattern" -- "$@" 2>/dev/null |
        sed "s#${allowed_literal}##g" |
        grep -EI "$pattern" || true)
    if [[ -n "$matches" ]]; then
        echo "ERROR: $description"
        echo "$matches"
        fail=1
    fi
}

forbidden_files=$(
    find . -type f -print | sed 's#^\./##' |
    grep -E '(^|/)(\.env($|\.)|coverage\.out$|config\.ya?ml$|.*\.(pem|key|p12|pfx)$|outbox/|dist/)' || true
    true
)
if [[ -n "$forbidden_files" ]]; then
    echo "ERROR: generated state or credential-bearing files are present:"
    echo "$forbidden_files"
    fail=1
fi

if [[ -n "$(find presets -type f -print -quit 2>/dev/null || true)" ]]; then
    echo "ERROR: model preset workflows must remain in the coordinator repository"
    fail=1
fi

report_matches "a live credential-shaped value is present" \
    '(sk_live_[A-Za-z0-9]{16,}|sk_test_[A-Za-z0-9]{16,}|pk_live_[A-Za-z0-9]{16,}|sbp_[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|postgres(ql)?://[^[:space:]/]+:[^[:space:]@]+@|https://[^/[:space:]]+:[^/[:space:]]+@)' \
    .

report_matches "a local absolute path is present" \
    '/Users/[^/[:space:]]+/' \
    . ':(exclude)scripts/audit-public.sh'

report_matches "coordinator-only policy leaked into public agent source" \
    '(PLATFORM_FEE_PERCENT|TARGET_MARGIN|gpu_owner_share|settlement|stripe_transfer|payout_status|load-balancer|queue-dispatcher|applyComfyGenerationOptions|isKnownDockerModel|knownWorkspaces|ltx-video-0\.9|wan-2\.1-|sdxl-base|stable-diffusion-1\.5)' \
    '*.go' '*.sh' '*.ps1' ':(exclude)scripts/audit-public.sh' \
    ':(exclude)*_integration_test.go' ':(exclude)*_e2e_test.go' \
    ':(exclude)bootstrap_test.go' ':(exclude)full_pipeline_test.go' \
    ':(exclude)media_pipeline_test.go' ':(exclude)preset_workflows_test.go' \
    ':(exclude)hardening_test.go' ':(exclude)executor_test.go'

report_matches_except_literal "private coordinator configuration leaked into public agent source" \
    '(POOL_API_SECRET|JWT_SECRET|DATABASE_URL|REDIS_URL|SUPABASE_SERVICE_ROLE_KEY|SUPABASE_ACCESS_TOKEN|rungpu-504008|420004393585|gjwmnwdlbsgyucnjgqck|\.run\.app)' \
    'https://rungpu-pool-api-420004393585.us-central1.run.app' \
    . ':(exclude)scripts/audit-public.sh'

rungpu_contacts=$(git grep --no-index -hEo '[A-Za-z0-9._%+-]+@rungpu\.io' -- . 2>/dev/null | sort -u || true)
if [[ "$rungpu_contacts" != "support@rungpu.io" ]]; then
    echo "ERROR: public contact addresses must use support@rungpu.io exclusively"
    echo "$rungpu_contacts"
    fail=1
fi

module_path=$(awk '$1 == "module" { print $2; exit }' go.mod)
if [[ "$module_path" == "github.com/RunGPU-io/rungpu-agent" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit_emails=$(git log -1 --format='%ae%n%ce' | sort -u)
    if [[ "$commit_emails" != "support@rungpu.io" ]]; then
        echo "ERROR: public commit metadata must use support@rungpu.io exclusively"
        fail=1
    fi
fi

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
