#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

die() {
  echo "$*" >&2
  exit 2
}

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT" "$1" ;;
  esac
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got: $value"
}

pid_is_running() {
  local pid=$1
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r /proc/$pid/stat ]]; then
    local state
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
    [[ $state != Z ]] || return 1
  fi
}

read_pid() {
  [[ -f $PID_FILE ]] || return 1
  local pid extra
  read -r pid extra < "$PID_FILE" || return 1
  [[ $pid =~ ^[1-9][0-9]*$ && -z ${extra:-} ]] || return 1
  printf '%s\n' "$pid"
}

read_owner() {
  [[ -f $OWNER_FILE ]] || return 1
  local owner extra
  read -r owner extra < "$OWNER_FILE" || return 1
  [[ $owner =~ ^[a-zA-Z0-9._-]+$ && -z ${extra:-} ]] || return 1
  printf '%s\n' "$owner"
}

write_atomic() {
  local destination=$1
  local value=$2
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv -f "$temporary" "$destination"
}

remove_marker_if_owned() {
  local path=$1
  local owner=$2
  local recorded=
  [[ -f $path ]] || return 0
  read -r recorded < "$path" || return 0
  [[ $recorded == "$owner" ]] && rm -f "$path"
}

controller_is_owned() {
  local pid=$1
  local owner=$2
  pid_is_running "$pid" || return 1
  [[ -r /proc/$pid/stat && -r /proc/$pid/cmdline ]] || return 1

  local process_group session
  read -r process_group session < <(awk '{print $5, $6}' "/proc/$pid/stat") || return 1
  [[ $process_group == "$pid" && $session == "$pid" ]] || return 1

  local argument saw_script=0 saw_action=0 saw_owner=0
  while IFS= read -r -d '' argument; do
    [[ $argument == "$ROOT/scripts/launch_7b_ssh_common.sh" ]] && saw_script=1
    [[ $argument == __controller ]] && saw_action=1
    [[ $argument == "$owner" ]] && saw_owner=1
  done < "/proc/$pid/cmdline"
  (( saw_script && saw_action && saw_owner ))
}

remove_controller_state() {
  local pid=$1
  local owner=$2
  remove_marker_if_owned "$PID_FILE" "$pid"
  remove_marker_if_owned "$READY_FILE" "$owner"
  remove_marker_if_owned "$OWNER_FILE" "$owner"
}

: "${MODEL_NAME:?entry script must set MODEL_NAME}"
: "${HOSTFILE:?entry script must set HOSTFILE}"
: "${MODEL_LAUNCHER:?entry script must set MODEL_LAUNCHER}"
: "${CONFIG_FILE:?entry script must set CONFIG_FILE}"
: "${OUTPUT_DIR:?entry script must set OUTPUT_DIR}"
: "${LOG_DIR:?entry script must set LOG_DIR}"
: "${PID_FILE:?entry script must set PID_FILE}"

HOSTFILE=$(absolute_path "$HOSTFILE")
MODEL_LAUNCHER=$(absolute_path "$MODEL_LAUNCHER")
CONFIG_FILE=$(absolute_path "$CONFIG_FILE")
OUTPUT_DIR=$(absolute_path "$OUTPUT_DIR")
LOG_DIR=$(absolute_path "$LOG_DIR")
PID_FILE=$(absolute_path "$PID_FILE")
READY_FILE="${PID_FILE}.ready"
OWNER_FILE="${PID_FILE}.owner"
CONTROLLER_LOG="$LOG_DIR/${MODEL_NAME}_controller.log"

MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PYTHON_BIN=$(absolute_path "${PYTHON_BIN:-$ROOT/.venv-train/bin/python}")
TRAIN_TOKENS_FILE=$(absolute_path "${TRAIN_TOKENS_FILE:-$ROOT/data/fineweb_100BT_pretok/train.bin}")
VAL_TOKENS_FILE=$(absolute_path "${VAL_TOKENS_FILE:-$ROOT/data/fineweb_100BT_pretok/val.bin}")
TRAIN_ITERS=${TRAIN_ITERS:-40000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
WARMUP_ITERS=${WARMUP_ITERS:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
EVAL_ITERS=${EVAL_ITERS:-20}
SSH_BIN=${SSH_BIN:-ssh}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-10}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-30}
DRY_RUN=${DRY_RUN:-0}

declare -a HOSTS=()
declare -A SEEN=()

validate_configuration() {
  [[ -r $HOSTFILE ]] || die "hostfile not readable: $HOSTFILE"
  HOSTS=()
  SEEN=()
  local raw line host extra
  while IFS= read -r raw || [[ -n $raw ]]; do
    line=${raw%%#*}
    read -r host extra <<< "$line"
    [[ -z ${host:-} ]] && continue
    [[ -z ${extra:-} ]] || die "hostfile line must contain one host: $raw"
    [[ -z ${SEEN[$host]+x} ]] || die "duplicate host: $host"
    SEEN[$host]=1
    HOSTS+=("$host")
  done < "$HOSTFILE"
  (( ${#HOSTS[@]} == 4 )) || die "hostfile must contain exactly 4 unique hosts"
  MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}

  local setting
  for setting in MASTER_PORT GPUS_PER_NODE TRAIN_ITERS MICRO_BATCH_SIZE \
    GLOBAL_BATCH_SIZE SAVE_INTERVAL EVAL_INTERVAL EVAL_ITERS \
    SSH_CONNECT_TIMEOUT STARTUP_TIMEOUT; do
    require_positive_integer "$setting" "${!setting}"
  done
  [[ $WARMUP_ITERS =~ ^[0-9]+$ ]] || die "WARMUP_ITERS must be a nonnegative integer, got: $WARMUP_ITERS"
  [[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die "DRY_RUN must be 0 or 1"
  (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * 16) == 0 )) || \
    die "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * 16"

  [[ -x $PYTHON_BIN ]] || die "Python executable not found: $PYTHON_BIN"
  [[ -x $MODEL_LAUNCHER ]] || die "model launcher not executable: $MODEL_LAUNCHER"
  [[ -r $CONFIG_FILE ]] || die "config not readable: $CONFIG_FILE"
  [[ ${TRAIN_TOKENS_FILE##*/} == train.bin ]] || \
    die "training tokens must be named train.bin: $TRAIN_TOKENS_FILE"
  [[ ${VAL_TOKENS_FILE##*/} == val.bin ]] || \
    die "validation tokens must be named val.bin: $VAL_TOKENS_FILE"
  [[ -r $TRAIN_TOKENS_FILE ]] || die "training tokens not found: $TRAIN_TOKENS_FILE"
  [[ -r $VAL_TOKENS_FILE ]] || die "validation tokens not found: $VAL_TOKENS_FILE"
  if [[ -e $OUTPUT_DIR && ! -d $OUTPUT_DIR ]]; then
    die "output path is not a directory: $OUTPUT_DIR"
  fi
  if [[ -d $OUTPUT_DIR ]] && [[ -n $(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    die "output directory must be empty: $OUTPUT_DIR"
  fi
}

print_contract() {
  echo "MODEL_NAME=$MODEL_NAME"
  echo "HOSTFILE=$HOSTFILE"
  echo "MODEL_LAUNCHER=$MODEL_LAUNCHER"
  echo "CONFIG_FILE=$CONFIG_FILE"
  local rank
  for rank in "${!HOSTS[@]}"; do
    echo "rank $rank: ${HOSTS[$rank]}"
  done
  echo "NNODES=4"
  echo "GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "MASTER_ADDR=$MASTER_ADDR"
  echo "MASTER_PORT=$MASTER_PORT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "TRAIN_TOKENS_FILE=$TRAIN_TOKENS_FILE"
  echo "VAL_TOKENS_FILE=$VAL_TOKENS_FILE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "PID_FILE=$PID_FILE"
  echo "OWNER_FILE=$OWNER_FILE"
  echo "CONTROLLER_LOG=$CONTROLLER_LOG"
  echo "TRAIN_ITERS=$TRAIN_ITERS"
  echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
  echo "WARMUP_ITERS=$WARMUP_ITERS"
  echo "SAVE_INTERVAL=$SAVE_INTERVAL"
  echo "EVAL_INTERVAL=$EVAL_INTERVAL"
  echo "EVAL_ITERS=$EVAL_ITERS"
  echo "MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0"
}

REMOTE_SCRIPT=$(cat <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

mode=$1
shift

if [[ $mode == preflight ]]; then
  project=$1
  python_bin=$2
  model_launcher=$3
  config_file=$4
  train_tokens_file=$5
  val_tokens_file=$6

  [[ -d $project ]] || { echo "project directory not found: $project" >&2; exit 2; }
  [[ -x $python_bin ]] || { echo "Python executable not found: $python_bin" >&2; exit 2; }
  [[ -x $model_launcher ]] || { echo "model launcher not executable: $model_launcher" >&2; exit 2; }
  [[ -r $config_file ]] || { echo "config not readable: $config_file" >&2; exit 2; }
  [[ -r $train_tokens_file ]] || { echo "training tokens not found: $train_tokens_file" >&2; exit 2; }
  [[ -r $val_tokens_file ]] || { echo "validation tokens not found: $val_tokens_file" >&2; exit 2; }
  "$python_bin" -c \
    'import torch; import megatron.core; import transformer_engine.pytorch; import triton'
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  (( gpu_count == 8 )) || { echo "expected 8 GPUs, found $gpu_count" >&2; exit 2; }
  exit 0
fi

[[ $mode == launch ]] || { echo "unknown remote mode: $mode" >&2; exit 2; }
project=$1
python_bin=$2
model_launcher=$3
config_file=$4
train_tokens_file=$5
val_tokens_file=$6
output_dir=$7
model_name=$8
node_rank=$9
shift 9
nnodes=$1
master_addr=$2
master_port=$3
gpus_per_node=$4
train_iters=$5
micro_batch_size=$6
global_batch_size=$7
warmup_iters=$8
save_interval=$9
shift 9
eval_interval=$1
eval_iters=$2

cd "$project"
unset PYTHONPATH
site_packages=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
torch_lib=$("$python_bin" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')
nvidia_libs=
if [[ -d $site_packages/nvidia ]]; then
  while IFS= read -r library_dir; do
    [[ -z $nvidia_libs ]] && nvidia_libs=$library_dir || nvidia_libs="$nvidia_libs:$library_dir"
  done < <(find "$site_packages/nvidia" -type d -name lib -print | sort)
fi
clean_ld=/usr/local/cuda-13.2/lib64
[[ -z $nvidia_libs ]] || clean_ld="$clean_ld:$nvidia_libs"
clean_ld="$clean_ld:$torch_lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=$CUDA_HOME
export LD_LIBRARY_PATH="$clean_ld"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0
export MEGATRON_ENABLE_LR_SCHEDULER=1
export MEGATRON_LR_WARMUP_ITERS="$warmup_iters"
export MEGATRON_LR_DECAY_ITERS="$train_iters"
export MEGATRON_LR_DECAY_STYLE=cosine
export MEGATRON_LR_WARMUP_INIT_LR=0

exec env \
  NNODES="$nnodes" \
  NODE_RANK="$node_rank" \
  MASTER_ADDR="$master_addr" \
  MASTER_PORT="$master_port" \
  GPUS_PER_NODE="$gpus_per_node" \
  PYTHON_BIN="$python_bin" \
  CONFIG_FILE="$config_file" \
  TRAIN_TOKENS_FILE="$train_tokens_file" \
  VAL_TOKENS_FILE="$val_tokens_file" \
  OUTPUT_DIR="$output_dir" \
  TRAIN_ITERS="$train_iters" \
  MICRO_BATCH_SIZE="$micro_batch_size" \
  GLOBAL_BATCH_SIZE="$global_batch_size" \
  WARMUP_ITERS="$warmup_iters" \
  SAVE_INTERVAL="$save_interval" \
  EVAL_INTERVAL="$eval_interval" \
  EVAL_ITERS="$eval_iters" \
  MODEL_NAME="$model_name" \
  "$model_launcher"
REMOTE
)

remote_command() {
  local command argument quoted
  command='bash -s --'
  for argument in "$@"; do
    printf -v quoted '%q' "$argument"
    command+=" $quoted"
  done
  printf '%s\n' "$command"
}

run_remote() {
  local host=$1
  shift
  local command
  command=$(remote_command "$@")
  "$SSH_BIN" \
    -o BatchMode=yes \
    -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
    "root@$host" "$command" <<< "$REMOTE_SCRIPT"
}

declare -a ACTIVE_PIDS=()
CONTROLLER_OWNER=

controller_cleanup() {
  local pid
  trap - EXIT TERM INT
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  remove_marker_if_owned "$READY_FILE" "$CONTROLLER_OWNER"
}

run_controller() {
  validate_configuration
  print_contract
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  trap controller_cleanup EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT

  local rank host
  for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    echo "preflight rank $rank: $host"
    run_remote "$host" preflight "$ROOT" "$PYTHON_BIN" "$MODEL_LAUNCHER" \
      "$CONFIG_FILE" "$TRAIN_TOKENS_FILE" "$VAL_TOKENS_FILE"
  done

  for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    local node_log="$LOG_DIR/${MODEL_NAME}_node${rank}.log"
    local command
    command=$(remote_command launch "$ROOT" "$PYTHON_BIN" "$MODEL_LAUNCHER" \
      "$CONFIG_FILE" "$TRAIN_TOKENS_FILE" "$VAL_TOKENS_FILE" "$OUTPUT_DIR" \
      "$MODEL_NAME" "$rank" 4 "$MASTER_ADDR" "$MASTER_PORT" "$GPUS_PER_NODE" \
      "$TRAIN_ITERS" "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$WARMUP_ITERS" \
      "$SAVE_INTERVAL" "$EVAL_INTERVAL" "$EVAL_ITERS")
    echo "starting node rank $rank on $host; log=$node_log"
    "$SSH_BIN" \
      -o BatchMode=yes \
      -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
      "root@$host" "$command" <<< "$REMOTE_SCRIPT" > "$node_log" 2>&1 &
    ACTIVE_PIDS[$rank]=$!
  done

  local startup_probe
  for ((startup_probe=0; startup_probe<5; startup_probe++)); do
    sleep 0.1
    for rank in "${!ACTIVE_PIDS[@]}"; do
      pid_is_running "${ACTIVE_PIDS[$rank]}" || {
        echo "node rank $rank (${HOSTS[$rank]}) exited during startup" >&2
        return 1
      }
    done
  done
  write_atomic "$READY_FILE" "$CONTROLLER_OWNER"

  local failed=0
  for rank in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$rank]}"; then
      echo "node rank $rank: OK"
    else
      echo "node rank $rank (${HOSTS[$rank]}): FAILED"
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  echo "overall exit code: $failed"
  return "$failed"
}

start_controller() {
  command -v setsid >/dev/null || die "setsid is required to supervise the controller"
  mkdir -p "$LOG_DIR" "$(dirname "$PID_FILE")"

  local existing_pid= existing_owner=
  existing_pid=$(read_pid 2>/dev/null) || true
  existing_owner=$(read_owner 2>/dev/null) || true
  if [[ -n $existing_pid ]]; then
    if [[ -n $existing_owner ]] && controller_is_owned "$existing_pid" "$existing_owner"; then
      die "$MODEL_NAME controller already running with PID $existing_pid"
    fi
    remove_marker_if_owned "$PID_FILE" "$existing_pid"
  fi
  [[ -z $existing_owner ]] || {
    remove_marker_if_owned "$READY_FILE" "$existing_owner"
    remove_marker_if_owned "$OWNER_FILE" "$existing_owner"
  }
  [[ ! -e $PID_FILE ]] || rm -f "$PID_FILE"
  [[ ! -e $READY_FILE ]] || rm -f "$READY_FILE"
  [[ ! -e $OWNER_FILE ]] || rm -f "$OWNER_FILE"

  validate_configuration
  print_contract

  local owner="${MODEL_NAME}.$$.$RANDOM.$RANDOM"
  write_atomic "$OWNER_FILE" "$owner"
  nohup setsid "$0" __controller "$owner" > "$CONTROLLER_LOG" 2>&1 < /dev/null &
  local controller_pid=$!
  write_atomic "$PID_FILE" "$controller_pid"

  local attempts=$(( STARTUP_TIMEOUT * 10 ))
  local attempt ready_owner=
  for ((attempt=0; attempt<attempts; attempt++)); do
    if [[ -f $READY_FILE ]]; then
      read -r ready_owner < "$READY_FILE" || true
      if [[ $ready_owner == "$owner" ]] && controller_is_owned "$controller_pid" "$owner"; then
        echo "$MODEL_NAME controller RUNNING with PID $controller_pid"
        return 0
      fi
    fi
    if ! pid_is_running "$controller_pid"; then
      wait "$controller_pid" 2>/dev/null || true
      remove_controller_state "$controller_pid" "$owner"
      echo "$MODEL_NAME controller failed during startup; see $CONTROLLER_LOG" >&2
      return 1
    fi
    sleep 0.1
  done

  if controller_is_owned "$controller_pid" "$owner"; then
    kill -TERM -- "-$controller_pid" 2>/dev/null || true
  fi
  wait "$controller_pid" 2>/dev/null || true
  remove_controller_state "$controller_pid" "$owner"
  echo "$MODEL_NAME controller did not become ready within ${STARTUP_TIMEOUT}s" >&2
  return 1
}

show_status() {
  local pid= owner=
  pid=$(read_pid 2>/dev/null) || true
  owner=$(read_owner 2>/dev/null) || true
  if [[ -n $pid && -n $owner ]] && controller_is_owned "$pid" "$owner"; then
    echo "$MODEL_NAME controller RUNNING with PID $pid"
    return 0
  fi
  echo "$MODEL_NAME controller STOPPED"
  return 1
}

follow_logs() {
  [[ -f $CONTROLLER_LOG ]] || die "controller log not found: $CONTROLLER_LOG"
  exec tail -n 100 -f "$CONTROLLER_LOG"
}

stop_controller() {
  local pid= owner=
  pid=$(read_pid 2>/dev/null) || true
  owner=$(read_owner 2>/dev/null) || true
  if [[ -z $pid || -z $owner ]]; then
    echo "$MODEL_NAME controller STOPPED"
    [[ -z $pid ]] || remove_marker_if_owned "$PID_FILE" "$pid"
    [[ -z $owner ]] || {
      remove_marker_if_owned "$READY_FILE" "$owner"
      remove_marker_if_owned "$OWNER_FILE" "$owner"
    }
    return 0
  fi
  if ! controller_is_owned "$pid" "$owner"; then
    echo "$MODEL_NAME controller STOPPED"
    remove_controller_state "$pid" "$owner"
    return 0
  fi

  kill -TERM -- "-$pid"
  local attempt
  for ((attempt=0; attempt<300; attempt++)); do
    if ! controller_is_owned "$pid" "$owner"; then
      remove_controller_state "$pid" "$owner"
      echo "$MODEL_NAME controller STOPPED"
      return 0
    fi
    sleep 0.1
  done
  echo "failed to stop $MODEL_NAME controller PID $pid after 30 seconds" >&2
  return 1
}

action=${1:-start}
case "$action" in
  start)
    if [[ $DRY_RUN == 1 ]]; then
      validate_configuration
      print_contract
      echo "dry run: no SSH sessions started"
      exit 0
    fi
    start_controller
    ;;
  status) show_status ;;
  logs) follow_logs ;;
  stop) stop_controller ;;
  __controller)
    CONTROLLER_OWNER=${2:-direct.$$}
    [[ $CONTROLLER_OWNER =~ ^[a-zA-Z0-9._-]+$ ]] || die "invalid controller owner token"
    run_controller
    ;;
  *) die "usage: $0 [start|status|logs|stop]" ;;
esac
