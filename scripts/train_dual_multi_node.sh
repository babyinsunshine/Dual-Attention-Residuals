#!/usr/bin/env bash
set -euo pipefail
: "${NNODES:?set NNODES}"
: "${NODE_RANK:?set NODE_RANK}"
: "${MASTER_ADDR:?set MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT:-29500}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
args=(--config configs/7b_moe_dual.yaml)
[[ -n ${TRAIN_TOKENS_FILE:-} ]] && args+=(--train-tokens-file "$TRAIN_TOKENS_FILE")
[[ -n ${VAL_TOKENS_FILE:-} ]] && args+=(--val-tokens-file "$VAL_TOKENS_FILE")
[[ -n ${OUTPUT_DIR:-} ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -n ${TRAIN_ITERS:-} ]] && args+=(--train-iters "$TRAIN_ITERS")
cmd=("$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="${GPUS_PER_NODE:-8}" --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" train.py "${args[@]}")
[[ ${DRY_RUN:-0} == 1 ]] && { printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; }
exec "${cmd[@]}"
