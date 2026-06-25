#!/usr/bin/env bash
# run.sh — SaaS-Bench concurrent evaluation launcher
# Usage: ./scripts/run.sh [options]
#
# Examples:
#   ./scripts/run.sh                                       # run with default settings
#   ./scripts/run.sh --workers 5                           # 5 concurrent workers
#   ./scripts/run.sh --no-isolation                        # do not start Docker isolation containers
#   ./scripts/run.sh --task-ids task_ids.txt
#   ./scripts/run.sh --tasks-dir tasks/Business                 # run a single domain only

set -euo pipefail

# -- Path setup --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# -- Load .env (if present) so LLM_API_KEY/LLM_BASE_URL/LLM_MODEL get exported.
# This parser intentionally does not source .env as shell code.  It supports
# KEY=value plus multi-line list syntax for LLM_API_KEYS/GEMINI_API_KEYS:
#   GEMINI_API_KEYS=[
#     "key1",
#     "key2"
#   ]
_trim_env_text() {
    local value="$1"
    value="${value//$'\r'/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

_strip_env_scalar() {
    local value
    value="$(_trim_env_text "$1")"
    if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
            value="${value:1:${#value}-2}"
        fi
    fi
    printf '%s' "$value"
}

_normalize_env_list() {
    local raw part
    raw="${1//$'\r'/}"
    raw="${raw//$'\n'/,}"
    raw="${raw//[/,}"
    raw="${raw//]/,}"
    raw="${raw//(/,}"
    raw="${raw//)/,}"
    raw="${raw//\"/}"
    raw="${raw//\'/}"
    raw="${raw//;/,}"

    local parts=()
    IFS=',' read -r -a parts <<< "$raw"
    local cleaned=()
    for part in "${parts[@]}"; do
        part="$(_trim_env_text "$part")"
        [[ -n "$part" ]] && cleaned+=("$part")
    done
    local IFS=,
    printf '%s' "${cleaned[*]}"
}

_export_env_assignment() {
    local key="$1"
    local value="$2"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 0

    if [[ "$key" == "LLM_API_KEYS" || "$key" == "GEMINI_API_KEYS" ]]; then
        value="$(_normalize_env_list "$value")"
    else
        value="$(_strip_env_scalar "$value")"
    fi

    printf -v "$key" '%s' "$value"
    export "$key"
}

_load_env_file() {
    local env_file="$1"
    local line trimmed key value collecting_key collecting_value
    collecting_key=""
    collecting_value=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line//$'\r'/}"

        if [[ -n "$collecting_key" ]]; then
            collecting_value+=$'\n'"$line"
            if [[ "$line" == *"]"* || "$line" == *")"* ]]; then
                _export_env_assignment "$collecting_key" "$collecting_value"
                collecting_key=""
                collecting_value=""
            fi
            continue
        fi

        trimmed="$(_trim_env_text "$line")"
        [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
        [[ "$trimmed" == export\ * ]] && trimmed="${trimmed#export }"
        [[ "$trimmed" == *"="* ]] || continue

        key="$(_trim_env_text "${trimmed%%=*}")"
        value="${trimmed#*=}"

        if [[ "$key" == "LLM_API_KEYS" || "$key" == "GEMINI_API_KEYS" ]]; then
            if [[ "$value" == *"["* || "$value" == *"("* ]]; then
                if [[ "$value" != *"]"* && "$value" != *")"* ]]; then
                    collecting_key="$key"
                    collecting_value="$value"
                    continue
                fi
            fi
        fi

        _export_env_assignment "$key" "$value"
    done < "$env_file"

    if [[ -n "$collecting_key" ]]; then
        _export_env_assignment "$collecting_key" "$collecting_value"
    fi
}

if [[ -f "$REPO_ROOT/.env" ]]; then
    _load_env_file "$REPO_ROOT/.env"
fi

# -- Python interpreter ------------------------------------------------------
PYTHON="${PYTHON:-python3}"
export BROWSER_USE_LOGGING_LEVEL=warning

# -- Default arguments -------------------------------------------------------
TASKS_DIR="${REPO_ROOT}/tasks"
MODEL="${LLM_MODEL}"
WORKERS=8
MAX_STEPS=400
HOSTNAME_VAL="localhost"
RESULT_DIR="${REPO_ROOT}/results-subset-v1"
APPS_YAML="${REPO_ROOT}/saas_bench/apps.yaml"
NO_ISOLATION=""
TASK_IDS_FILE=""
LOG_FILE="modify-gfull.log"
RERUN_EXISTING=""

# -- Argument parsing --------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --tasks-dir <path>      Task directory root (default: rollout/tasks)
  --model <name>          LLM model name (default: claude-opus-4-6)
  --workers <n>           Number of concurrent workers (default: 3)
  --max-steps <n>         Max steps per task (default: 400)
  --hostname <host>       Hostname the agent uses to access apps (default: localhost)
  --result-dir <path>     Output directory for results (default: rollout/results)
  --apps-yaml <path>      Path to apps.yaml (default: rollout/apps.yaml)
  --no-isolation          Disable Docker container isolation; connect directly to already-running apps via fixed_port
  --task-ids <path>       Path to a file containing task ids to run
  --rerun-existing        Re-run tasks even if their result JSON files already exist
  --log <file>            Also write output to a log file
  -h, --help              Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks-dir)   TASKS_DIR="$2";       shift 2 ;;
        --model)       MODEL="$2";           shift 2 ;;
        --workers)     WORKERS="$2";         shift 2 ;;
        --max-steps)   MAX_STEPS="$2";       shift 2 ;;
        --hostname)    HOSTNAME_VAL="$2";    shift 2 ;;
        --result-dir)  RESULT_DIR="$2";      shift 2 ;;
        --apps-yaml)   APPS_YAML="$2";       shift 2 ;;
        --no-isolation) NO_ISOLATION="--no-isolation"; shift ;;
        --rerun-existing) RERUN_EXISTING="--rerun-existing"; shift ;;
        --task-ids)    TASK_IDS_FILE="$2";   shift 2 ;;
        --log)         LOG_FILE="$2";        shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# -- Dependency check --------------------------------------------------------
if ! "$PYTHON" -c "import saas_bench" 2>/dev/null; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    if ! "$PYTHON" -c "import saas_bench" 2>/dev/null; then
        echo "[ERROR] saas_bench module not found; run from the repo root, or pip install -e . first" >&2
        exit 1
    fi
fi

if [[ -z "${LLM_BASE_URL:-}" ]]; then
    echo "[ERROR] LLM_BASE_URL not set; please cp .env.example .env, fill it in, and retry" >&2
    exit 1
fi

if [[ -z "${LLM_API_KEY:-}" && -z "${LLM_API_KEYS:-}" && -z "${GEMINI_API_KEYS:-}" ]]; then
    echo "[ERROR] Set LLM_API_KEY, or for Gemini set LLM_API_KEYS/GEMINI_API_KEYS in .env" >&2
    exit 1
fi

if [[ ! -d "$TASKS_DIR" ]]; then
    echo "[ERROR] Task directory does not exist: $TASKS_DIR" >&2
    exit 1
fi

if [[ ! -f "$APPS_YAML" ]]; then
    echo "[ERROR] apps.yaml does not exist: $APPS_YAML" >&2
    exit 1
fi

if [[ -n "$TASK_IDS_FILE" && ! -f "$TASK_IDS_FILE" ]]; then
    echo "[ERROR] Task ids file does not exist: $TASK_IDS_FILE" >&2
    exit 1
fi

if [[ -n "$TASK_IDS_FILE" ]]; then
    TASK_IDS_FILE="$(cd "$(dirname "$TASK_IDS_FILE")" && pwd)/$(basename "$TASK_IDS_FILE")"
fi

# -- Print configuration summary ---------------------------------------------
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
TASK_COUNT="$(find "$TASKS_DIR" -name "meta.json" | wc -l | tr -d ' ')"
MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
MODEL_RESULT_DIR="${RESULT_DIR}/${MODEL_SLUG}"

mask_env_key() {
    local value="${1:-}"
    if [[ -z "$value" ]]; then
        echo "unset"
    else
        echo "set (masked, len=${#value})"
    fi
}

echo "============================================"
echo "  SaaS-Bench concurrent evaluation"
echo "============================================"
echo "  Start time   : $TIMESTAMP"
echo "  Task dir     : $TASKS_DIR"
echo "  Found tasks  : $TASK_COUNT"
echo "  Model        : $MODEL"
echo "  LLM base URL : ${LLM_BASE_URL:-unset}"
echo "  API keys     : LLM_API_KEY=$(mask_env_key "${LLM_API_KEY:-}") | MINDRA_API_KEY=$(mask_env_key "${MINDRA_API_KEY:-}")"
echo "  Workers      : $WORKERS"
echo "  Max steps    : $MAX_STEPS"
echo "  Hostname     : $HOSTNAME_VAL"
echo "  Result dir   : $MODEL_RESULT_DIR"
echo "  Isolation    : ${NO_ISOLATION:-enabled (Docker per-slot)}"
[[ -n "$TASK_IDS_FILE" ]] && echo "  Task ids file: $TASK_IDS_FILE"
echo "============================================"
echo ""

# -- Build command -----------------------------------------------------------
CMD=(
    "$PYTHON" -m saas_bench.run
    --tasks-dir   "$TASKS_DIR"
    --task-ids   "task_ids_21.txt"
    --model       "$MODEL"
    --workers     "$WORKERS"
    --max-steps   "$MAX_STEPS"
    --hostname    "$HOSTNAME_VAL"
    --result-dir  "$RESULT_DIR"
    --apps-yaml   "$APPS_YAML"
)

[[ -n "$NO_ISOLATION" ]] && CMD+=("$NO_ISOLATION")
[[ -n "$RERUN_EXISTING" ]] && CMD+=("$RERUN_EXISTING")

[[ -n "$TASK_IDS_FILE" ]] && CMD+=(--task-ids "$TASK_IDS_FILE")

# -- Execute ----------------------------------------------------------------
cd "$REPO_ROOT"

if [[ -n "$LOG_FILE" ]]; then
    echo "Also writing log to: $LOG_FILE"
    mkdir -p "$(dirname "$LOG_FILE")"
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${CMD[@]}"
fi

EXIT_CODE=${PIPESTATUS[0]:-$?}

echo ""
echo "============================================"
echo "  Done, exit code: $EXIT_CODE"
echo "  Result dir: $MODEL_RESULT_DIR"
[[ -f "$MODEL_RESULT_DIR/summary.json" ]] && echo "  Summary:" && "$PYTHON" -c "
import json
s = json.load(open('$MODEL_RESULT_DIR/summary.json'))
print(f\"    Total : {s.get('completed',0)}/{s.get('total',0)} completed\")
for cat, ds in sorted(s.get('domains', {}).items()):
    avg = ds.get('avg_verify_score', 0)
    print(f\"    {cat:6s}: {ds.get('completed',0)}/{ds.get('total',0)} completed  verify avg={avg:.3f}\")
"
echo "============================================"

exit $EXIT_CODE
