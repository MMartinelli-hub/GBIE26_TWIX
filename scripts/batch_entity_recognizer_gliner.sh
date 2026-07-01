#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_entity_recognizer.sh"
TRAIN_CONFIG="$SCRIPT_DIR/configs/entityRecognizer/gliner_train.yaml"
INFER_CONFIG="$SCRIPT_DIR/configs/entityRecognizer/gliner_inference.yaml"

LOG_DIR="$SCRIPT_DIR/logs/entityRecognizer/batch_gliner"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints/entityRecognizer"
RUNS_DIR="$PROJECT_DIR/runs/entityRecognizer"

mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR" "$RUNS_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_entity_recognizer_gliner.sh

What it does:
  - Iterates over a hardcoded list of GLiNER model names
  - For each model:
      1. fine-tunes the GLiNER model
      2. runs inference with the fine-tuned checkpoint

Expected layout:
  - scripts/run_entity_recognizer.sh
  - scripts/configs/entityRecognizer/gliner_train.yaml
  - scripts/configs/entityRecognizer/gliner_inference.yaml
  - run_entity_recognizer.py in the project root

Derived output paths:
  checkpoint:  checkpoints/entityRecognizer/<derived_model_name>
  predictions: runs/entityRecognizer/<derived_model_name>_inference_predictions.json
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
# Define GLiNER model names here
# ============================================================
MODELS=(
  "numind/NuNerZero"
  "numind/NuNerZero_long_context"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_gliner_${TIMESTAMP}.log"

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
log "Batch GLiNER entity recognizer run started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "Train config:     $TRAIN_CONFIG"
log "Inference config: $INFER_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for MODEL_NAME in "${MODELS[@]}"; do
    OUTPUT="${MODEL_NAME##*/}"
    OUTPUT="${OUTPUT// /_}"
    OUTPUT="${OUTPUT//:/_}"

    CHECKPOINT_PATH="checkpoints/entityRecognizer/$OUTPUT"
    PREDICTIONS_PATH="runs/entityRecognizer/${OUTPUT}_inference_predictions.json"

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
            "gliner.model_name=$MODEL_NAME" \
            "gliner.output_dir=$CHECKPOINT_PATH"

    run_step \
        "[INFER] model=$MODEL_NAME output=$OUTPUT" \
        bash "$RUNNER_SCRIPT" \
        --config "$INFER_CONFIG" \
        --override \
            "gliner.model_path=$CHECKPOINT_PATH" \
            "gliner.inference_output_path=$PREDICTIONS_PATH"
done

log ""
log "=================================================="
log "All GLiNER entity recognizer experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
