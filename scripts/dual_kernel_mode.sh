#!/usr/bin/env bash

configure_dual_kernel_mode() {
  local mode=${DUAL_KERNEL_MODE:-fused}
  local optimized
  case "$mode" in
    fused) optimized=1 ;;
    reference) optimized=0 ;;
    *) echo "DUAL_KERNEL_MODE must be fused or reference, got: $mode" >&2; return 2 ;;
  esac

  export DUAL_KERNEL_MODE="$mode"
  # Both modes implement cross-key/self-value retrieval and transport the
  # current partial state; only the execution kernels differ.
  export MEGATRON_DUAL_KV_MODE=cross
  export MEGATRON_DUAL_AB_SOURCE=attended
  export MEGATRON_DUAL_KEEP_SOURCE=partial
  export MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL=1
  export MEGATRON_DUAL_CURRENT_KEY_IMPL=stream_norm
  export MEGATRON_DUAL_DEFER_STREAM_MIX=1
  export MEGATRON_DUAL_ATTRES_LIMIT=0
  export MEGATRON_DUAL_ATTRES_LIKE_FIRST_ATTN=0
  export MEGATRON_DUAL_EXCLUDE_RAW_HISTORY=0
  export MEGATRON_DUAL_RAWLESS_RESIDUAL=0
  export MEGATRON_DUAL_EMBED_AS_PARTIAL=0
  export MEGATRON_DUAL_CACHE_HISTORY_LOGITS=0
  export MEGATRON_DUAL_CACHE_HISTORY_PREFIX="$optimized"
  export MEGATRON_DUAL_CACHE_HISTORY_PREFIX_PAIR="$optimized"
  export MEGATRON_DUAL_RECOMPUTE_HISTORY_VALUES="$optimized"
  export MEGATRON_DUAL_RECOMPUTE_HISTORY_KEYS="$optimized"
  export MEGATRON_DUAL_TRITON_HISTORY_PREFIX_PAIR="$optimized"
  export MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM="$optimized"
  export MEGATRON_DUAL_FUSE_PARTIAL_KEEP_MIX="$optimized"
  export MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ="$optimized"
  export MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ_STRICT="$optimized"
  export MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK="$optimized"
  export MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK_STRICT="$optimized"
  export MEGATRON_DUAL_TE_ATTENTION_EPILOGUE="$optimized"
}
