#!/bin/bash
#
# Run all tests for the Tokenize GPU Agent.
#
# Usage:
#   ./scripts/test.sh              # run all tests
#   ./scripts/test.sh -v           # verbose output
#   ./scripts/test.sh -race        # with race detector
#   ./scripts/test.sh -run Ollama  # filter by test name
#   ./scripts/test.sh -cover       # with coverage report
#   ./scripts/test.sh -short       # skip slow tests
#   ./scripts/test.sh -integration # only integration tests
#
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Parse flags ──────────────────────────────────────────────────────────────
VERBOSE=""
RACE=""
RUN=()
COVER=""
SHORT=""
INTEGRATION=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -race|--race)
            RACE="-race"
            shift
            ;;
        -run|--run)
            RUN=(-run "$2")
            shift 2
            ;;
        -cover|--cover)
            COVER="-coverprofile=coverage.out"
            shift
            ;;
        -short|--short)
            SHORT="-short"
            shift
            ;;
        -integration|--integration)
            INTEGRATION="true"
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/test.sh [flags]"
            echo ""
            echo "Flags:"
            echo "  -v, --verbose       Verbose test output"
            echo "  -race, --race       Enable Go race detector"
            echo "  -run, --run REGEX   Run only tests matching regex"
            echo "  -cover, --cover     Generate coverage report"
            echo "  -short, --short     Skip slow tests"
            echo "  -integration        Run only integration tests"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  ./scripts/test.sh                    # all tests"
            echo "  ./scripts/test.sh -v                 # verbose"
            echo "  ./scripts/test.sh -race              # race detector"
            echo "  ./scripts/test.sh -run Ollama        # filter by name"
            echo "  ./scripts/test.sh -cover             # coverage report"
            echo "  ./scripts/test.sh -v -race           # verbose + race"
            echo "  ./scripts/test.sh -integration       # integration only"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If --integration, filter to integration test names
if [[ "$INTEGRATION" == "true" && ${#RUN[@]} -eq 0 ]]; then
    RUN=(-run 'TestFullOllamaLifecycle|TestConcurrentJob|TestServerRecovery|TestServerFlapping|TestExecutorFullPipeline_WithProgress|TestExecutorWithUpload_FullChain|TestExecutorWithCustomFiles_LoRA|TestE2E_|TestFullPipeline_|TestFullLifecycle|TestHuggingFace|TestCustomDocker|TestWellKnownModel|TestOllamaModel|TestE2E_RealAgent_')
fi

# ── Header ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  Tokenize GPU Agent — Test Suite${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Check Go ─────────────────────────────────────────────────────────────────
if ! command -v go &> /dev/null; then
    echo -e "${RED}Error: Go is not installed.${NC}"
    echo "  Install from: https://go.dev/dl/"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
echo -e "${CYAN}Go:${NC}       $GO_VERSION"
echo -e "${CYAN}Dir:${NC}      $(pwd)"

# Show what we're running
FLAGS="$VERBOSE $RACE ${RUN[*]:-} $COVER $SHORT ${EXTRA_ARGS[*]:-}"
FLAGS=$(echo "$FLAGS" | xargs)  # trim whitespace
if [[ -n "$FLAGS" ]]; then
    echo -e "${CYAN}Flags:${NC}    $FLAGS"
fi
echo ""

# ── Count tests ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}Discovering tests...${NC}"
TEST_COUNT=$(go test ./... -list '.*' 2>/dev/null | grep -c '^Test' || true)
echo -e "Found ${BOLD}${TEST_COUNT}${NC} tests across all packages"
echo ""

# ── Run tests ────────────────────────────────────────────────────────────────
echo -e "${YELLOW}Running tests...${NC}"
echo ""

START_TIME=$(date +%s)

# Build the command
# Note: 120s is enough for all tests EXCEPT TestE2E_RealAgent_OllamaInference
# which may pull a 2GB model. Run that separately with -timeout 600s.
CMD=(go test ./... -timeout 120s)
[[ -n "$VERBOSE" ]] && CMD+=("$VERBOSE")
[[ -n "$RACE" ]] && CMD+=("$RACE")
if [[ ${#RUN[@]} -gt 0 ]]; then
    CMD+=("${RUN[@]}")
fi
[[ -n "$COVER" ]] && CMD+=("$COVER")
[[ -n "$SHORT" ]] && CMD+=("$SHORT")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

# Run it
if "${CMD[@]}"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  ✅ ALL TESTS PASSED  (${ELAPSED}s)${NC}"
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════${NC}"
else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo -e "${RED}${BOLD}══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}${BOLD}  ❌ TESTS FAILED  (${ELAPSED}s)${NC}"
    echo -e "${RED}${BOLD}══════════════════════════════════════════════════════════${NC}"
    exit 1
fi

# ── Coverage report ──────────────────────────────────────────────────────────
if [[ -n "$COVER" ]]; then
    echo ""
    echo -e "${YELLOW}Coverage report:${NC}"
    go tool cover -func=coverage.out | tail -1
    echo ""
    echo -e "Full report: ${CYAN}go tool cover -html=coverage.out${NC}"
fi
