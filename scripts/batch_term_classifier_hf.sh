#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_term_classifier.sh"
TRAIN_CONFIG="$SCRIPT_DIR/configs/termClassifier/hf_train.yaml"
INFER_CONFIG="$SCRIPT_DIR/configs/termClassifier/hf_inference.yaml"

LOG_DIR="$SCRIPT_DIR/logs/termClassifier/batch_e2e"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints/termClassifier"
RUNS_DIR="$PROJECT_DIR/runs/termClassifier"

mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR" "$RUNS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_term_classifier_hf.sh

What it does:
  - Iterates over a hardcoded list of model names
  - For each model:
      1. derives output name as the last token after "/"
      2. trains the term classifier
      3. runs inference with the trained checkpoint

Expected layout:
  - scripts/run_term_classifier.sh
  - scripts/configs/termClassifier/hf_train.yaml
  - scripts/configs/termClassifier/hf_inference.yaml
  - run_term_classifier.py (or run_term_classification.py, depending on your setup) in the project root

Derived output paths:
  checkpoint: checkpoints/termClassifier/<derived_model_name>
  predictions: runs/termClassifier/<derived_model_name>.json
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

if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "[ERROR] Train config not found: $TRAIN_CONFIG" >&2
    exit 1
fi

if [[ ! -f "$INFER_CONFIG" ]]; then
    echo "[ERROR] Inference config not found: $INFER_CONFIG" >&2
    exit 1
fi

# ============================================================
# Define model names here
# ============================================================
MODELS=(
    "michiyasunaga/BioLinkBERT-base"
    "michiyasunaga/BioLinkBERT-large"
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    "microsoft/BiomedNLP-BiomedElectra-base-uncased-abstract"
    "microsoft/BiomedNLP-BiomedElectra-large-uncased-abstract"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_e2e_${TIMESTAMP}.log"

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
log "Batch E2E term classifier run started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "Train config:     $TRAIN_CONFIG"
log "Inference config: $INFER_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for MODEL_NAME in "${MODELS[@]}"; do
    OUTPUT="${MODEL_NAME##*/}"

    CHECKPOINT_PATH="checkpoints/termClassifier/$OUTPUT"
    PREDICTIONS_PATH="runs/termClassifier/${OUTPUT}.json"

    log ""
    log "=================================================="
    log "Experiment"
    log "  model_name:  $MODEL_NAME"
    log "  output:      $OUTPUT"
    log "  checkpoint:  $CHECKPOINT_PATH"
    log "  predictions: $PREDICTIONS_PATH"
    log "=================================================="

    run_step \
        "[TRAIN] model=$MODEL_NAME output=$OUTPUT" \
        bash "$RUNNER_SCRIPT" \
        --config "$TRAIN_CONFIG" \
        --override \
            "hf.model_name=$MODEL_NAME" \
            "hf.output_dir=$CHECKPOINT_PATH"

    run_step \
        "[INFER] model=$MODEL_NAME output=$OUTPUT" \
        bash "$RUNNER_SCRIPT" \
        --config "$INFER_CONFIG" \
        --override \
            "hf.model_path=$CHECKPOINT_PATH" \
            "hf.inference_output_path=$PREDICTIONS_PATH"
done

log ""
log "=================================================="
log "All experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="