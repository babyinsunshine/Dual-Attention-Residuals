#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MODEL_NAME=attnres
export HOSTFILE=${HOSTFILE:-$ROOT/hostfile_attnres}
export MODEL_LAUNCHER=$ROOT/scripts/train_attnres_multi_node.sh
export CONFIG_FILE=$ROOT/configs/7b_moe_attnres.yaml
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/7b_moe_attnres_4node_40k}
export LOG_DIR=${LOG_DIR:-$ROOT/logs/7b_moe_attnres_4node_40k}
export PID_FILE=${PID_FILE:-$LOG_DIR/controller.pid}
exec "$ROOT/scripts/launch_7b_ssh_common.sh" "${1:-start}"
