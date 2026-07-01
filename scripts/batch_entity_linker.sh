#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

RUNNER_SCRIPT="$SCRIPT_DIR/run_entity_linker.sh"
INFER_CONFIG="$SCRIPT_DIR/configs/entityLinker/inference.yaml"
LOG_DIR="$SCRIPT_DIR/logs/entityLinker/batch_inference"

mkdir -p "$LOG_DIR"

usage() {
    cat <<USAGE
Usage:
  bash scripts/batch_entity_linker.sh

What it does:
  - Iterates over hardcoded entity-linker inference experiments
  - Keeps all YAML parameters fixed except:
      linker.retriever_id
      linker.reranker_id
      linker.retriever_model_name_or_path
      linker.reranker_model_name_or_path
      linker.retriever_index_dir
  - Runs scripts/run_entity_linker.sh with scripts/configs/entityLinker/inference.yaml

Edit ENTITY_LINKER_EXPERIMENTS near the top of this file to change the grid.
Each row has this format:
  "retriever_id|reranker_id|retriever_model_name_or_path|reranker_model_name_or_path|retriever_index_dir"

Use "null" for reranker_id / reranker_model_name_or_path when running
retriever-only inference.
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
# Define inference experiments here.
# Format:
#   "retriever_id|reranker_id|retriever_model_name_or_path|reranker_model_name_or_path|retriever_index_dir"
#
# Notes:
#   - For trained retrievers/rerankers, you may pass the auto-named run folder
#     or its explicit component subfolder.
#   - For BM25, use null model/index paths unless you have a saved BM25 index.
#   - For PRIOR, retriever_model_name_or_path must be the mention_counter.json
#     file, and entity_dict_path in inference.yaml should point at the
#     PRIOR-compatible dictionary if you generated one.
# ============================================================
ENTITY_LINKER_EXPERIMENTS=(
    # Retriever-only runs from batch_train_retriever.sh
    #"textembedding|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=intfloat__e5-base|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=intfloat__e5-base"
    #"textembedding|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|null|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"e5bm25|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=intfloat__e5-base|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=intfloat__e5-base"
    #"e5bm25|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=michiyasunaga__BioLinkBERT-base|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"e5bm25|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|null|runs/entityLinker/retriever_checkpoint_retriever=e5bm25_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract"
    #"dualencoder|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=intfloat__e5-base|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=intfloat__e5-base"
    #"dualencoder|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=michiyasunaga__BioLinkBERT-base|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"dualencoder|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|null|runs/entityLinker/retriever_checkpoint_retriever=dualencoder_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract"

    # Retriever + reranker runs from batch_train_reranker.sh
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=crossencoder_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=crossencoder_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|fevry|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=fevry_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|fevry|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=fevry_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|extend|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=extend_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|extend|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=extend_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|fusioned|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=fusioned_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    #"textembedding|fusioned|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=fusioned_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"

    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=crossencoder_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base_reranker=crossencoder_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=michiyasunaga__BioLinkBERT-base"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract_reranker=crossencoder_reranker_model=michiyasunaga__BioLinkBERT-base|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract"
    "textembedding|crossencoder|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_reranker_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract_reranker=crossencoder_reranker_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract|runs/entityLinker/retriever_checkpoint_retriever=textembedding_retriever_model=microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="$LOG_DIR/batch_entity_linker_${TIMESTAMP}.log"

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
log "Batch entity-linker inference started"
log "Timestamp:        $TIMESTAMP"
log "Project root:     $PROJECT_DIR"
log "Runner script:    $RUNNER_SCRIPT"
log "Inference config: $INFER_CONFIG"
log "Master log:       $MASTER_LOG"
log "=================================================="

for EXPERIMENT in "${ENTITY_LINKER_EXPERIMENTS[@]}"; do
    IFS="|" read -r RETRIEVER_ID RERANKER_ID RETRIEVER_MODEL_NAME_OR_PATH RERANKER_MODEL_NAME_OR_PATH RETRIEVER_INDEX_DIR <<< "$EXPERIMENT"

    log ""
    log "=================================================="
    log "Experiment"
    log "  retriever_id:                 $RETRIEVER_ID"
    log "  reranker_id:                  $RERANKER_ID"
    log "  retriever_model_name_or_path: $RETRIEVER_MODEL_NAME_OR_PATH"
    log "  reranker_model_name_or_path:  $RERANKER_MODEL_NAME_OR_PATH"
    log "  retriever_index_dir:          $RETRIEVER_INDEX_DIR"
    log "=================================================="

    run_step \
        "[INFER ENTITY LINKER] retriever=$RETRIEVER_ID reranker=$RERANKER_ID" \
        bash "$RUNNER_SCRIPT" \
        --config "$INFER_CONFIG" \
        --override \
            "linker.retriever_id=$RETRIEVER_ID" \
            "linker.reranker_id=$RERANKER_ID" \
            "linker.retriever_model_name_or_path=$RETRIEVER_MODEL_NAME_OR_PATH" \
            "linker.reranker_model_name_or_path=$RERANKER_MODEL_NAME_OR_PATH" \
            "linker.retriever_index_dir=$RETRIEVER_INDEX_DIR"
done

log ""
log "=================================================="
log "All entity-linker inference experiments completed successfully."
log "Master log saved to: $MASTER_LOG"
log "=================================================="
