#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

CONFIG_DIR="$SCRIPT_DIR/configs/termExtractor"
LOG_DIR="$SCRIPT_DIR/logs/termExtractor"
RUNS_DIR="$PROJECT_DIR/runs/termExtractor"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints/termExtractor"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RUNS_DIR" "$CHECKPOINTS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/run_term_extractor.sh
  bash scripts/run_term_extractor.sh --config scripts/configs/termExtractor/hf_train.yaml
  bash scripts/run_term_extractor.sh --config scripts/configs/termExtractor/hf_train.yaml --override hf.num_epochs=10 hf.batch_size=16

Notes:
  - The script assumes it lives in PROJECT_ROOT/scripts/
  - run_term_extractor.py must be located in PROJECT_ROOT/
  - configs are expected in scripts/configs/termExtractor/
  - logs are saved in scripts/logs/termExtractor/
  - runs are expected under runs/termExtractor/
  - checkpoints are expected under checkpoints/termExtractor/
USAGE
}

CONFIG_FILE=""
OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -lt 2 ]] && { echo "[ERROR] Missing value after --config" >&2; exit 1; }
            CONFIG_FILE="$2"
            shift 2
            ;;
        --override)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                OVERRIDES+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$CONFIG_DIR/hf_train.yaml"
    echo "[INFO] No --config provided. Falling back to default: $CONFIG_FILE"
fi

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" 2>/dev/null && pwd)/$(basename "$CONFIG_FILE")"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/run_term_extractor.py" ]]; then
    echo "[ERROR] run_term_extractor.py not found in project root: $PROJECT_DIR" >&2
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG_BASENAME="$(basename "$CONFIG_FILE")"
CONFIG_STEM="${CONFIG_BASENAME%.*}"
LOG_FILE="$LOG_DIR/${CONFIG_STEM}_${TIMESTAMP}.log"

CMD=(python -u "$PROJECT_DIR/run_term_extractor.py" --config "$CONFIG_FILE")
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    CMD+=(--override)
    for ov in "${OVERRIDES[@]}"; do
        CMD+=("$ov")
    done
fi

{
    echo "=================================================="
    echo "Timestamp:       $TIMESTAMP"
    echo "Project root:    $PROJECT_DIR"
    echo "Config dir:      $CONFIG_DIR"
    echo "Log dir:         $LOG_DIR"
    echo "Runs dir:        $RUNS_DIR"
    echo "Checkpoints dir: $CHECKPOINTS_DIR"
    echo "Config file:     $CONFIG_FILE"
    echo "Log file:        $LOG_FILE"
    echo "Python script:   $PROJECT_DIR/run_term_extractor.py"
    if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
        echo "Overrides:"
        for ov in "${OVERRIDES[@]}"; do
            echo "  - $ov"
        done
    else
        echo "Overrides:       none"
    fi
    echo "Command:"
    printf '  %q' "${CMD[@]}"
    echo
    echo "=================================================="
    echo
} | tee "$LOG_FILE"

cd "$PROJECT_DIR"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo >> "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

exit "$EXIT_CODE"