#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/attnres_kernel_mode.sh"
configure_attnres_kernel_mode
echo "ATTRES_KERNEL_MODE=$ATTRES_KERNEL_MODE" >&2
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
args=(--config configs/7b_moe_attnres.yaml)
[[ -n ${TRAIN_TOKENS_FILE:-} ]] && args+=(--train-tokens-file "$TRAIN_TOKENS_FILE")
[[ -n ${VAL_TOKENS_FILE:-} ]] && args+=(--val-tokens-file "$VAL_TOKENS_FILE")
[[ -n ${OUTPUT_DIR:-} ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -n ${TRAIN_ITERS:-} ]] && args+=(--train-iters "$TRAIN_ITERS")
cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${GPUS_PER_NODE:-8}" train.py "${args[@]}")
[[ ${DRY_RUN:-0} == 1 ]] && { printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; }
exec "${cmd[@]}"
