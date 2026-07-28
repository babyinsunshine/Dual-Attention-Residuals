#!/usr/bin/env bash

configure_attnres_kernel_mode() {
  local mode=${ATTRES_KERNEL_MODE:-fused}
  local optimized

  case "$mode" in
    fused)
      optimized=1
      ;;
    reference)
      optimized=0
      ;;
    *)
      echo "ATTRES_KERNEL_MODE must be fused or reference, got: $mode" >&2
      return 2
      ;;
  esac

  export ATTRES_KERNEL_MODE="$mode"
  export MEGATRON_ATTRES_TRITON_WEIGHTED_SUM="$optimized"
  export MEGATRON_ATTRES_CACHE_HISTORY_LOGITS_PAIR=0
  export MEGATRON_ATTRES_CACHE_HISTORY_PREFIX_PAIR="$optimized"
  export MEGATRON_ATTRES_RECOMPUTE_HISTORY_VALUES="$optimized"
  export MEGATRON_ATTRES_RECOMPUTE_HISTORY_KEYS="$optimized"
  export MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_PAIR="$optimized"
  export MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM="$optimized"
  export MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK="$optimized"
  export MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT="$optimized"
  export MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE="$optimized"
  export MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_STRICT="$optimized"
  export MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE="$optimized"
}
