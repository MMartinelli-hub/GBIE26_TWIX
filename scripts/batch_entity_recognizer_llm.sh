#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_entity_recognizer.sh"
INFER_CONFIG="$SCRIPT_DIR/configs/entityRecognizer/llm_inference.yaml"

LOG_DIR="$SCRIPT_DIR/logs/entityRecognizer/batch_llm"
RUNS_DIR="$PROJECT_DIR/runs/entityRecognizer"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_entity_recognizer_llm.sh

What it does:
  - Iterates over a hardcoded list of LLM provider+model combinations
  - For each combination:
      1. runs inference with the specified LLM

Expected layout:
  - scripts/run_entity_recognizer.sh
  - scripts/configs/entityRecognizer/llm_inference.yaml
  - run_entity_recognizer.py in the project root

Notes:
  - LLMs are inference-only; no training occurs.
  - Adjust the PROVIDERS and MODELS arrays to match your available models.

Derived output paths:
  predictions: runs/entityRecognizer/<provider>_<model>_dev_predictions.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    echo "[ERROR] Runner script not found: $RUNNER_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$INFER_CONFIG" ]]; then
    echo "[ERROR] Inference config not found: $INFER_CONFIG" >&2
    exit 1
fi

# ============================================================
# Define provider+model combinations here
# ============================================================
# Format: "provider:model"
COMBINATIONS=(
    #"lmstudio:medgemma-4b-it-mlx"
    #"lmstudio:gemma-3-12b-it"
    "lmstudio:medgemma-27b-text-it-mlx"
    "lmstudio:gemma-3-27b-it"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_llm_${TIMESTAMP}.log"

log() {
    echo "$1" | tee -a "$MASTER_LOG"
}

run_step() {
    local description="$1"
    shift

    log ""
    log "--------------------------------------------------"
    log "$description"
    log "Command:"
    printf '  %q' "$@" | tee -a "$MASTER_LOG"
    echo | tee -a "$MASTER_LOG"
    log "--------------------------------------------------"

    "$@" 2>&1 | tee -a "$MASTER_LOG"
    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -ne 0 ]]; then
        log "[ERROR] Step failed with exit code $exit_code"
        exit "$exit_code"
    fi
}

log "=================================================="
log "Batch LLM entity recognizer run started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "Inference config: $INFER_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for COMBO in "${COMBINATIONS[@]}"; do
    IFS=':' read -r PROVIDER MODEL <<< "$COMBO"

    # Sanitize output name (replace special chars)
    OUTPUT="${PROVIDER}_${MODEL}"
    OUTPUT="${OUTPUT// /_}"
    OUTPUT="${OUTPUT//:/_}"
    OUTPUT="${OUTPUT////_}"

    PREDICTIONS_PATH="runs/entityRecognizer/${OUTPUT}_dev_predictions.json"

    log ""
    log "=================================================="
    log "Experiment"
    log "  provider:    $PROVIDER"
    log "  model:       $MODEL"
    log "  output:      $OUTPUT"
    log "  predictions: $PREDICTIONS_PATH"
    log "=================================================="

    run_step \
        "[INFER] provider=$PROVIDER model=$MODEL" \
        bash "$RUNNER_SCRIPT" \
        --config "$INFER_CONFIG" \
        --override \
            "llm.provider=$PROVIDER" \
            "llm.model=$MODEL" \
            "llm.inference_output_path=$PREDICTIONS_PATH"
done

log ""
log "=================================================="
log "All LLM entity recognizer experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
