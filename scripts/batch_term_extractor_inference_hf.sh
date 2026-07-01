#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_term_extractor.sh"
INFER_CONFIG="$SCRIPT_DIR/configs/termExtractor/hf_inference.yaml"

LOG_DIR="$SCRIPT_DIR/logs/termExtractor/batch_inference_hf"
RUNS_DIR="$PROJECT_DIR/runs/termExtractor"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_term_extractor_inference_hf.sh

What it does:
  - Iterates over a hardcoded list of model checkpoint paths
  - For each checkpoint:
      1. runs inference with the trained model

Expected layout:
  - scripts/run_term_extractor.sh
  - scripts/configs/termExtractor/hf_inference.yaml
  - run_term_extractor.py in the project root
  - Pre-trained checkpoints in checkpoints/termExtractor/

Notes:
  - This script performs inference only; training is skipped.
  - Ensure checkpoint paths point to existing trained models.

Derived output paths:
  predictions: runs/termExtractor/<derived_model_name>_dev_predictions.json
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
# Define checkpoint paths here
# ============================================================
CHECKPOINTS=(
    "checkpoints/termExtractor/BioLinkBERT-base"
    "checkpoints/termExtractor/BioLinkBERT-large"
    "checkpoints/termExtractor/BiomedNLP-BiomedBERT-base-uncased-abstract"
    "checkpoints/termExtractor/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    "checkpoints/termExtractor/BiomedNLP-BiomedBERT-large-uncased-abstract"
    "checkpoints/termExtractor/BiomedNLP-BiomedElectra-base-uncased-abstract"
    "checkpoints/termExtractor/BiomedNLP-BiomedElectra-large-uncased-abstract"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_inference_hf_${TIMESTAMP}.log"

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
log "Batch HF term extractor inference run started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "Inference config: $INFER_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
    # Extract model name from checkpoint path
    OUTPUT="${CHECKPOINT_PATH##*/}"
    OUTPUT="${OUTPUT// /_}"
    OUTPUT="${OUTPUT//:/_}"

    PREDICTIONS_PATH="runs/termExtractor/${OUTPUT}_dev_predictions.json"

    if [[ ! -d "$PROJECT_DIR/$CHECKPOINT_PATH" ]]; then
        log "[WARN] Checkpoint not found: $CHECKPOINT_PATH (skipping)"
        continue
    fi

    log ""
    log "=================================================="
    log "Experiment"
    log "  checkpoint:  $CHECKPOINT_PATH"
    log "  output:      $OUTPUT"
    log "  predictions: $PREDICTIONS_PATH"
    log "=================================================="

    run_step \
        "[INFER] checkpoint=$CHECKPOINT_PATH output=$OUTPUT" \
        bash "$RUNNER_SCRIPT" \
        --config "$INFER_CONFIG" \
        --override \
            "hf.model_path=$CHECKPOINT_PATH" \
            "hf.inference_output_path=$PREDICTIONS_PATH"
done

log ""
log "=================================================="
log "All HF term extractor inference experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
