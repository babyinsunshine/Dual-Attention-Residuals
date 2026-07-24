#!/usr/bin/env bash
set -euo pipefail

PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

HOSTS_VALUE="${HOSTS:-10.244.57.135 10.244.77.160 10.244.64.90 10.244.82.236}"
read -r -a HOST_ARRAY <<< "$HOSTS_VALUE"
(( ${#HOST_ARRAY[@]} > 0 )) || { echo "HOSTS must contain at least one host" >&2; exit 2; }

NNODES=${#HOST_ARRAY[@]}
MASTER_ADDR="${MASTER_ADDR:-${HOST_ARRAY[0]}}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT/.venv-train/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT/configs/7b_moe_baseline.yaml}"

TRAIN_TOKENS_FILE="${TRAIN_TOKENS_FILE:-$PROJECT/data/fineweb_100BT_pretok/smoke_snapshot/train.bin}"
VAL_TOKENS_FILE="${VAL_TOKENS_FILE:-$PROJECT/data/fineweb_100BT_pretok/smoke_snapshot/val.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/outputs/baseline_4node_60k}"
LOG_DIR="${LOG_DIR:-$PROJECT/logs/baseline_4node_60k}"

TRAIN_ITERS="${TRAIN_ITERS:-60000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_ITERS="${EVAL_ITERS:-20}"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_BIN="${SSH_BIN:-ssh}"
DRY_RUN="${DRY_RUN:-0}"

absolute_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT" "$1" ;;
  esac
}

PYTHON_BIN=$(absolute_project_path "$PYTHON_BIN")
CONFIG_FILE=$(absolute_project_path "$CONFIG_FILE")
TRAIN_TOKENS_FILE=$(absolute_project_path "$TRAIN_TOKENS_FILE")
VAL_TOKENS_FILE=$(absolute_project_path "$VAL_TOKENS_FILE")
OUTPUT_DIR=$(absolute_project_path "$OUTPUT_DIR")
LOG_DIR=$(absolute_project_path "$LOG_DIR")

require_positive_integer() {
  local name=$1
  local value=$2
  [[ $value =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  }
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  [[ $value =~ ^[0-9]+$ ]] || {
    echo "$name must be a nonnegative integer, got: $value" >&2
    exit 2
  }
}

for setting in MASTER_PORT GPUS_PER_NODE TRAIN_ITERS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SAVE_INTERVAL EVAL_INTERVAL EVAL_ITERS SSH_CONNECT_TIMEOUT; do
  require_positive_integer "$setting" "${!setting}"
done
require_nonnegative_integer WARMUP_ITERS "$WARMUP_ITERS"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

[[ -x $PYTHON_BIN ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f $CONFIG_FILE ]] || { echo "config not found: $CONFIG_FILE" >&2; exit 2; }
[[ -f $TRAIN_TOKENS_FILE ]] || { echo "training tokens not found: $TRAIN_TOKENS_FILE" >&2; exit 2; }
[[ -f $VAL_TOKENS_FILE ]] || { echo "validation tokens not found: $VAL_TOKENS_FILE" >&2; exit 2; }

print_settings() {
  echo "PROJECT=$PROJECT"
  for rank in "${!HOST_ARRAY[@]}"; do
    echo "rank $rank: ${HOST_ARRAY[$rank]}"
  done
  echo "NNODES=$NNODES"
  echo "GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "MASTER_ADDR=$MASTER_ADDR"
  echo "MASTER_PORT=$MASTER_PORT"
  echo "TRAIN_ITERS=$TRAIN_ITERS"
  echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
  echo "WARMUP_ITERS=$WARMUP_ITERS"
  echo "SAVE_INTERVAL=$SAVE_INTERVAL"
  echo "EVAL_INTERVAL=$EVAL_INTERVAL"
  echo "EVAL_ITERS=$EVAL_ITERS"
  echo "TRAIN_TOKENS_FILE=$TRAIN_TOKENS_FILE"
  echo "VAL_TOKENS_FILE=$VAL_TOKENS_FILE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "LOG_DIR=$LOG_DIR"
}

print_settings
if [[ $DRY_RUN == 1 ]]; then
  echo "dry run: no SSH sessions started"
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

REMOTE_SCRIPT=$(cat <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

mode=$1
shift

if [[ $mode == preflight ]]; then
  project=$1
  python_bin=$2
  config_file=$3
  train_tokens_file=$4
  val_tokens_file=$5

  [[ -d $project ]] || { echo "project directory not found: $project" >&2; exit 2; }
  [[ -x $python_bin ]] || { echo "Python executable not found: $python_bin" >&2; exit 2; }
  [[ -f $config_file ]] || { echo "config not found: $config_file" >&2; exit 2; }
  [[ -f $train_tokens_file ]] || { echo "training tokens not found: $train_tokens_file" >&2; exit 2; }
  [[ -f $val_tokens_file ]] || { echo "validation tokens not found: $val_tokens_file" >&2; exit 2; }
  exit 0
fi

[[ $mode == launch ]] || { echo "unknown remote mode: $mode" >&2; exit 2; }

project=$1
python_bin=$2
config_file=$3
train_tokens_file=$4
val_tokens_file=$5
output_dir=$6
node_rank=$7
nnodes=$8
master_addr=$9
shift 9
master_port=$1
gpus_per_node=$2
train_iters=$3
micro_batch_size=$4
global_batch_size=$5
warmup_iters=$6
save_interval=$7
eval_interval=$8
eval_iters=$9
shift 9
cuda_home=$1
nccl_debug=$2
nccl_debug_subsys=$3

cd "$project"
unset PYTHONPATH

site_packages=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
torch_lib=$("$python_bin" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')
nvidia_libs=""
if [[ -d $site_packages/nvidia ]]; then
  while IFS= read -r library_dir; do
    if [[ -z $nvidia_libs ]]; then
      nvidia_libs=$library_dir
    else
      nvidia_libs="$nvidia_libs:$library_dir"
    fi
  done < <(find "$site_packages/nvidia" -type d -name lib -print | sort)
fi

clean_ld="$cuda_home/lib64"
[[ -z $nvidia_libs ]] || clean_ld="$clean_ld:$nvidia_libs"
clean_ld="$clean_ld:$torch_lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

export CUDA_HOME="$cuda_home"
export CUDA_PATH="$cuda_home"
export LD_LIBRARY_PATH="$clean_ld"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

export NCCL_DEBUG="$nccl_debug"
export NCCL_DEBUG_SUBSYS="$nccl_debug_subsys"
export NCCL_IB_DISABLE=0

export MEGATRON_ENABLE_LR_SCHEDULER=1
export MEGATRON_LR_WARMUP_ITERS="$warmup_iters"
export MEGATRON_LR_DECAY_ITERS="$train_iters"
export MEGATRON_LR_DECAY_STYLE=cosine
export MEGATRON_LR_WARMUP_INIT_LR=0

exec "$python_bin" -m torch.distributed.run \
  --nnodes="$nnodes" \
  --nproc-per-node="$gpus_per_node" \
  --node-rank="$node_rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  train.py \
  --config "$config_file" \
  --train-tokens-file "$train_tokens_file" \
  --val-tokens-file "$val_tokens_file" \
  --output-dir "$output_dir" \
  --train-iters "$train_iters" \
  --micro-batch-size "$micro_batch_size" \
  --global-batch-size "$global_batch_size" \
  --save-interval "$save_interval" \
  --eval-interval "$eval_interval" \
  --eval-iters "$eval_iters"
REMOTE
)

run_remote() {
  local host=$1
  shift
  printf '%s\n' "$REMOTE_SCRIPT" | "$SSH_BIN" \
    -o BatchMode=yes \
    -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
    "root@$host" bash -s -- "$@"
}

for rank in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$rank]}
  echo "preflight rank $rank: $host"
  run_remote "$host" preflight "$PROJECT" "$PYTHON_BIN" "$CONFIG_FILE" \
    "$TRAIN_TOKENS_FILE" "$VAL_TOKENS_FILE"
done

declare -a PIDS=()
for rank in "${!HOST_ARRAY[@]}"; do
  host=${HOST_ARRAY[$rank]}
  log_file="$LOG_DIR/baseline_node${rank}.log"
  echo "starting node rank $rank on $host; log=$log_file"
  run_remote "$host" launch \
    "$PROJECT" "$PYTHON_BIN" "$CONFIG_FILE" "$TRAIN_TOKENS_FILE" "$VAL_TOKENS_FILE" \
    "$OUTPUT_DIR" "$rank" "$NNODES" "$MASTER_ADDR" "$MASTER_PORT" "$GPUS_PER_NODE" \
    "$TRAIN_ITERS" "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$WARMUP_ITERS" \
    "$SAVE_INTERVAL" "$EVAL_INTERVAL" "$EVAL_ITERS" "$CUDA_HOME" "$NCCL_DEBUG" \
    "$NCCL_DEBUG_SUBSYS" >"$log_file" 2>&1 &
  PIDS[$rank]=$!
done

failed=0
for rank in "${!HOST_ARRAY[@]}"; do
  if wait "${PIDS[$rank]}"; then
    echo "node rank $rank: OK"
  else
    echo "node rank $rank: FAILED"
    failed=1
  fi
done

echo "overall exit code: $failed"
exit "$failed"
