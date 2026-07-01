#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_relation_extractor.sh"
LLM_CONFIG="$SCRIPT_DIR/configs/relationExtractor/llm_inference.yaml"

LOG_DIR="$SCRIPT_DIR/logs/relationExtractor/batch_llm"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints/relationExtractor"
RUNS_DIR="$PROJECT_DIR/runs/relationExtractor"

mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR" "$RUNS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_relation_extractor_llm.sh

What it does:
  - Iterates over a hardcoded list of (provider, model) pairs
  - For each pair:
      1. derives output name from the model name
      2. overrides both llm.provider and llm.model
      3. runs relationExtractor inference
      4. stores predictions and checkpoint under relationExtractor subfolders

Expected layout:
  - scripts/run_relation_extractor.sh
  - scripts/configs/relationExtractor/llm_inference.yaml
  - run_relation_extractor.py in the project root

Derived output paths:
  predictions: runs/relationExtractor/<provider>_<derived_model_name>_dev_predictions.json
  checkpoint:  checkpoints/relationExtractor/<provider>_<derived_model_name>_dev_inference_checkpoint.json
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

if [[ ! -f "$LLM_CONFIG" ]]; then
    echo "[ERROR] LLM config not found: $LLM_CONFIG" >&2
    exit 1
fi

# ============================================================
# Define experiments here as: "provider|model"
# ============================================================
EXPERIMENTS=(
    "lmstudio|medgemma-4b-it-mlx"
    "lmstudio|gemma-3-12b-it"
    "lmstudio|medgemma-27b-text-it-mlx"
    "lmstudio|gemma-3-27b-it"
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
log "Batch LLM relation extractor inference started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "LLM config:       $LLM_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r PROVIDER MODEL_NAME <<< "$experiment"

    OUTPUT="${MODEL_NAME##*/}"
    OUTPUT="${OUTPUT// /_}"
    OUTPUT="${OUTPUT//:/_}"

    PREDICTIONS_PATH="runs/relationExtractor/${PROVIDER}_${OUTPUT}_dev_predictions.json"
    CHECKPOINT_PATH="checkpoints/relationExtractor/${PROVIDER}_${OUTPUT}_dev_inference_checkpoint.json"

    log ""
    log "=================================================="
    log "Experiment"
    log "  provider:    $PROVIDER"
    log "  model_name:  $MODEL_NAME"
    log "  output:      $OUTPUT"
    log "  predictions: $PREDICTIONS_PATH"
    log "  checkpoint:  $CHECKPOINT_PATH"
    log "=================================================="

    run_step \
        "[LLM INFER] provider=$PROVIDER model=$MODEL_NAME output=$OUTPUT" \
        bash "$RUNNER_SCRIPT" \
        --config "$LLM_CONFIG" \
        --override \
            "llm.provider=$PROVIDER" \
            "llm.model=$MODEL_NAME" \
            "llm.inference_output_path=$PREDICTIONS_PATH" \
            "llm.checkpoint_path=$CHECKPOINT_PATH"
done

log ""
log "=================================================="
log "All LLM relation extractor experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
