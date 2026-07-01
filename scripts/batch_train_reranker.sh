#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_entity_linker.sh"
TRAIN_CONFIG="$SCRIPT_DIR/configs/entityLinker/train_reranker.yaml"
LOG_DIR="$SCRIPT_DIR/logs/entityLinker/batch_train_reranker"

mkdir -p "$LOG_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_train_reranker.sh

What it does:
  - Iterates over hardcoded reranker experiments
  - Keeps all YAML parameters fixed except:
      linker.retriever_id
      linker.reranker_id
      linker.retriever_model_name_or_path
      linker.reranker_model_name_or_path
  - Runs scripts/run_entity_linker.sh with overrides

Edit the RERANKER_EXPERIMENTS array near the top of this file to change the
experiment grid. Each row has this format:
  "retriever_id|reranker_id|retriever_model_name_or_path|reranker_model_name_or_path"
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
# Define reranker experiments here.
# Format:
#   "retriever_id|reranker_id|retriever_model_name_or_path|reranker_model_name_or_path"
# ============================================================
RERANKER_EXPERIMENTS=(
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract "
    #"textembedding|chatel|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    #"textembedding|chatel|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"textembedding|fevry|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    #"textembedding|fevry|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"textembedding|extend|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    #"textembedding|extend|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"textembedding|fusioned|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    #"textembedding|fusioned|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint|michiyasunaga/BioLinkBERT-base"
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint|google-bert/bert-base-uncased"
    #"dualencoder|crossencoder|runs/entityLinker/retriever_checkpoint|michiyasunaga/BioLinkBERT-base"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|michiyasunaga/BioLinkBERT-base"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|michiyasunaga/BioLinkBERT-base"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_train_reranker_${TIMESTAMP}.log"

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
log "Batch entity-linker reranker training started"
log "Timestamp:     $TIMESTAMP"
log "Project root:  $PROJECT_DIR"
log "Runner script: $RUNNER_SCRIPT"
log "Train config:  $TRAIN_CONFIG"
log "Master log:    $MASTER_LOG"
log "=================================================="

for EXPERIMENT in "${RERANKER_EXPERIMENTS[@]}"; do
    IFS="|" read -r RETRIEVER_ID RERANKER_ID RETRIEVER_MODEL_NAME_OR_PATH RERANKER_MODEL_NAME_OR_PATH <<< "$EXPERIMENT"

    log ""
    log "=================================================="
    log "Experiment"
    log "  retriever_id:                 $RETRIEVER_ID"
    log "  reranker_id:                  $RERANKER_ID"
    log "  retriever_model_name_or_path: $RETRIEVER_MODEL_NAME_OR_PATH"
    log "  reranker_model_name_or_path:  $RERANKER_MODEL_NAME_OR_PATH"
    log "=================================================="

    run_step \
        "[TRAIN RERANKER] retriever=$RETRIEVER_ID reranker=$RERANKER_ID" \
        bash "$RUNNER_SCRIPT" \
        --config "$TRAIN_CONFIG" \
        --override \
            "linker.retriever_id=$RETRIEVER_ID" \
            "linker.reranker_id=$RERANKER_ID" \
            "linker.retriever_model_name_or_path=$RETRIEVER_MODEL_NAME_OR_PATH" \
            "linker.reranker_model_name_or_path=$RERANKER_MODEL_NAME_OR_PATH" \
            "linker.retriever_index_dir=$RETRIEVER_MODEL_NAME_OR_PATH" \
            "linker.train_retriever=false" \
            "linker.train_reranker=true"
done

log ""
log "=================================================="
log "All reranker training experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
