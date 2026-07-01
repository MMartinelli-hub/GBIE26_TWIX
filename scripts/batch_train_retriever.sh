#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_entity_linker.sh"
TRAIN_CONFIG="$SCRIPT_DIR/configs/entityLinker/train_retriever.yaml"
LOG_DIR="$SCRIPT_DIR/logs/entityLinker/batch_train_retriever"

mkdir -p "$LOG_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_train_retriever.sh

What it does:
  - Iterates over hardcoded retriever experiments
  - Keeps all YAML parameters fixed except:
      linker.retriever_id
      linker.retriever_model_name_or_path
  - Runs scripts/run_entity_linker.sh with overrides

Edit the RETRIEVER_EXPERIMENTS array near the top of this file to change the
experiment grid. Each row has this format:
  "retriever_id|retriever_model_name_or_path"
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

# ============================================================
# Define retriever experiments here.
# Format: "retriever_id|retriever_model_name_or_path"
# ============================================================
RETRIEVER_EXPERIMENTS=(
    #"textembedding|intfloat/e5-base"
    #"textembedding|michiyasunaga/BioLinkBERT-base"
    #"textembedding|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"e5bm25|intfloat/e5-base"
    #"e5bm25|michiyasunaga/BioLinkBERT-base"
    #"e5bm25|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"dualencoder|intfloat/e5-base"
    #"dualencoder|michiyasunaga/BioLinkBERT-base"
    #"dualencoder|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    "textembedding|michiyasunaga/BioLinkBERT-base"
    "textembedding|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_train_retriever_${TIMESTAMP}.log"

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
log "Batch entity-linker retriever training started"
log "Timestamp:     $TIMESTAMP"
log "Project root:  $PROJECT_DIR"
log "Runner script: $RUNNER_SCRIPT"
log "Train config:  $TRAIN_CONFIG"
log "Master log:    $MASTER_LOG"
log "=================================================="

for EXPERIMENT in "${RETRIEVER_EXPERIMENTS[@]}"; do
    IFS="|" read -r RETRIEVER_ID RETRIEVER_MODEL_NAME_OR_PATH <<< "$EXPERIMENT"

    log ""
    log "=================================================="
    log "Experiment"
    log "  retriever_id:                 $RETRIEVER_ID"
    log "  retriever_model_name_or_path: $RETRIEVER_MODEL_NAME_OR_PATH"
    log "=================================================="

    run_step \
        "[TRAIN RETRIEVER] retriever=$RETRIEVER_ID model=$RETRIEVER_MODEL_NAME_OR_PATH" \
        bash "$RUNNER_SCRIPT" \
        --config "$TRAIN_CONFIG" \
        --override \
            "linker.retriever_id=$RETRIEVER_ID" \
            "linker.retriever_model_name_or_path=$RETRIEVER_MODEL_NAME_OR_PATH" \
            "linker.reranker_id=null" \
            "linker.reranker_model_name_or_path=null" \
            "linker.train_retriever=true" \
            "linker.train_reranker=false"
done

log ""
log "=================================================="
log "All retriever training experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
