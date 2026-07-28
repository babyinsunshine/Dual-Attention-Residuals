from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Verified 7B MoE defaults. Explicit environment values remain authoritative,
# including setting any feature to 0 for reference or compatibility runs.
_RESIDUAL_ENV_DEFAULTS = {
    "MEGATRON_ATTRES_TRITON_WEIGHTED_SUM": "1",
    "MEGATRON_ATTRES_TRITON_WEIGHTED_SUM_BLOCK_H": "512",
    "MEGATRON_ATTRES_TRITON_WEIGHTED_SUM_NUM_WARPS": "4",
    "MEGATRON_ATTRES_CACHE_HISTORY_LOGITS_PAIR": "0",
    "MEGATRON_ATTRES_CACHE_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_ATTRES_RECOMPUTE_HISTORY_VALUES": "1",
    "MEGATRON_ATTRES_RECOMPUTE_HISTORY_KEYS": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_BLOCK_H": "64",
    "MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_NUM_WARPS": "4",
    "MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_BLOCK_H": "128",
    "MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_NUM_WARPS": "4",
    "MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_STRICT": "1",
    "MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE": "1",
    "MEGATRON_RESIDUAL_TE_EPILOGUE_BLOCK_H": "1024",
    "MEGATRON_DUAL_CACHE_HISTORY_PREFIX": "1",
    "MEGATRON_DUAL_CACHE_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_DUAL_RECOMPUTE_HISTORY_VALUES": "1",
    "MEGATRON_DUAL_RECOMPUTE_HISTORY_KEYS": "1",
    "MEGATRON_DUAL_RECOMPUTE_CURRENT_KEY_STREAMS": "1",
    "MEGATRON_DUAL_CACHE_HISTORY_LOGITS": "0",
    "MEGATRON_DUAL_SCALAR_MIX_IMPL": "rmsnorm",
    "MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ": "1",
    "MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ_STRICT": "1",
    "MEGATRON_DUAL_TRITON_APPLY_BLOCK_H": "1024",
    "MEGATRON_DUAL_TRITON_APPLY_NUM_WARPS": "8",
    # Use the same FP32 accumulator graph in the fused forward reconstruction
    # and full backward. Public tensors and returned gradients remain BF16.
    "MEGATRON_DUAL_TRITON_BF16_PARITY": "0",
    "MEGATRON_DUAL_TRITON_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_DUAL_TRITON_HISTORY_PREFIX_BLOCK_H": "64",
    "MEGATRON_DUAL_TRITON_HISTORY_PREFIX_NUM_WARPS": "4",
    "MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM": "1",
    "MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H": "128",
    "MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS": "4",
    "MEGATRON_DUAL_TE_ATTENTION_EPILOGUE": "1",
    "MEGATRON_DUAL_TE_EPILOGUE_BLOCK_H": "1024",
    "MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK": "1",
    "MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK_STRICT": "1",
    "MEGATRON_DUAL_TRITON_BACKWARD_POST": "1",
    "MEGATRON_DUAL_TRITON_BACKWARD_POST_BLOCK_H": "512",
    "MEGATRON_DUAL_TRITON_BACKWARD_POST_NUM_WARPS": "4",
    # This combination is parity checked against the PyTorch backward under
    # the public TP2/EP4 topology and avoids retaining the full autograd graph.
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL": "1",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_FINALIZE": "1",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_ROW_BLOCK_H": "512",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_ROW_NUM_WARPS": "4",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_POST_BLOCK_H": "512",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_POST_NUM_WARPS": "4",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_BLOCK_R": "1024",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_BLOCK_F": "8",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_NUM_WARPS": "4",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_BLOCK_R": "64",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_BLOCK_H": "64",
    "MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_NUM_WARPS": "4",
    "MEGATRON_DUAL_TRITON_BACKWARD_KEY_QUERY": "0",
    "MEGATRON_DUAL_CURRENT_KEY_IMPL": "stream_norm",
    "MEGATRON_DUAL_KV_MODE": "cross",
    "MEGATRON_DUAL_DEFER_STREAM_MIX": "1",
    "MEGATRON_DUAL_MERGE_IMPL": "lerp_addcmul",
    "MEGATRON_DUAL_HISTORY_IMPL": "einsum",
    "MEGATRON_DUAL_Q_INIT": "zero_antisym",
    "MEGATRON_DUAL_Q_INIT_EPS": "0",
    "MEGATRON_DUAL_Q_BALANCED": "1",
    "MEGATRON_DUAL_AB_INIT": "near_attres",
    "MEGATRON_DUAL_AB_SYMMETRY_BREAK_LOGIT": "0.05",
    "MEGATRON_DUAL_STATIC_RESIDUAL_LOGIT": "8.0",
    "MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL": "1",
    "MEGATRON_DUAL_AB_SOURCE": "attended",
    "MEGATRON_DUAL_KEEP_SOURCE": "partial",
    "MEGATRON_DUAL_FUSE_PARTIAL_KEEP_MIX": "1",
    "MEGATRON_DUAL_EXCLUDE_RAW_HISTORY": "0",
    "MEGATRON_DUAL_RAWLESS_RESIDUAL": "0",
    "MEGATRON_DUAL_EMBED_AS_PARTIAL": "0",
}
for _name, _value in _RESIDUAL_ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

try:
    from .attres_residual_triton import (
        paired_attres_history_logits,
        reuse_attres_history_logits,
        triton_attres_current_finalize,
        triton_attres_history_append_pack,
        triton_attres_weighted_sum,
        triton_is_available as _attres_residual_triton_is_available,
    )
except Exception:  # pragma: no cover - Triton is optional.
    paired_attres_history_logits = None
    reuse_attres_history_logits = None
    triton_attres_current_finalize = None
    triton_attres_history_append_pack = None
    triton_attres_weighted_sum = None

    def _attres_residual_triton_is_available() -> bool:
        return False


try:
    from .dual_residual_triton import (
        triton_attres_history_prefix_pair,
        triton_dual_deferred_merge,
        triton_dual_history_append_pack,
        triton_dual_history_key_pack_norm,
        triton_dual_te_attention_epilogue,
        triton_fixed_rms_norm,
        triton_fused_dual_prefix_residual_read,
        triton_history_prefix_pair,
        triton_te_attention_delta,
        triton_is_available as _dual_residual_triton_is_available,
    )
except Exception:  # pragma: no cover - Triton is optional.
    triton_attres_history_prefix_pair = None
    triton_dual_deferred_merge = None
    triton_dual_history_append_pack = None
    triton_dual_history_key_pack_norm = None
    triton_dual_te_attention_epilogue = None
    triton_fixed_rms_norm = None
    triton_fused_dual_prefix_residual_read = None
    triton_history_prefix_pair = None
    triton_te_attention_delta = None

    def _dual_residual_triton_is_available() -> bool:
        return False


_DUAL_PROFILE_STATS: dict[str, list[float]] = {}
_ATTRES_PROFILE_STATS: dict[str, list[float]] = {}
_DUAL_PROFILE_ON = os.environ.get("MEGATRON_DUAL_BLOCK_PROFILE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ATTRES_PROFILE_ON = os.environ.get("MEGATRON_ATTRES_BLOCK_PROFILE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _dual_profile_enabled() -> bool:
    return _DUAL_PROFILE_ON


def _dual_profile_now() -> float:
    if not _dual_profile_enabled():
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _dual_profile_record(name: str, start: float) -> None:
    if start <= 0.0:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_and_count = _DUAL_PROFILE_STATS.setdefault(name, [0.0, 0.0])
    total_and_count[0] += elapsed
    total_and_count[1] += 1.0


def pop_dual_profile_stats() -> dict[str, tuple[float, int, float]]:
    stats = {
        name: (values[0], int(values[1]), values[0] / max(1.0, values[1]))
        for name, values in _DUAL_PROFILE_STATS.items()
    }
    _DUAL_PROFILE_STATS.clear()
    return stats


def _attres_profile_enabled() -> bool:
    return _ATTRES_PROFILE_ON


def _attres_profile_now() -> float:
    if not _attres_profile_enabled():
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _attres_profile_record(name: str, start: float) -> None:
    if start <= 0.0:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_and_count = _ATTRES_PROFILE_STATS.setdefault(name, [0.0, 0.0])
    total_and_count[0] += elapsed
    total_and_count[1] += 1.0


def pop_attres_profile_stats() -> dict[str, tuple[float, int, float]]:
    stats = {
        name: (values[0], int(values[1]), values[0] / max(1.0, values[1]))
        for name, values in _ATTRES_PROFILE_STATS.items()
    }
    _ATTRES_PROFILE_STATS.clear()
    return stats


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


_DUAL_SPLIT_CURRENT_SOFTMAX = _env_bool("MEGATRON_DUAL_SPLIT_CURRENT_SOFTMAX", False)
_ATTRES_TRITON_WEIGHTED_SUM = _env_bool("MEGATRON_ATTRES_TRITON_WEIGHTED_SUM", False)
_ATTRES_CACHE_HISTORY_LOGITS_PAIR = _env_bool("MEGATRON_ATTRES_CACHE_HISTORY_LOGITS_PAIR", False)
_ATTRES_CACHE_HISTORY_PREFIX_PAIR = _env_bool("MEGATRON_ATTRES_CACHE_HISTORY_PREFIX_PAIR", False)
_ATTRES_RECOMPUTE_HISTORY_VALUES = _env_bool("MEGATRON_ATTRES_RECOMPUTE_HISTORY_VALUES", False)
_ATTRES_RECOMPUTE_HISTORY_KEYS = _env_bool("MEGATRON_ATTRES_RECOMPUTE_HISTORY_KEYS", False)
_ATTRES_TRITON_HISTORY_PREFIX_PAIR = _env_bool("MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_PAIR", False)
_ATTRES_TRITON_CURRENT_KEY_NORM = _env_bool("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM", False)
_ATTRES_TRITON_HISTORY_APPEND_PACK = _env_bool("MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK", False)
_ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT = _env_bool(
    "MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT", False
)
_ATTRES_TRITON_CURRENT_FINALIZE = _env_bool("MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE", False)
_ATTRES_TRITON_CURRENT_FINALIZE_STRICT = _env_bool(
    "MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_STRICT", False
)
_RESIDUAL_TE_ATTENTION_EPILOGUE = _env_bool("MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE", False)
_DUAL_TE_ATTENTION_EPILOGUE = _env_bool("MEGATRON_DUAL_TE_ATTENTION_EPILOGUE", False)
_DUAL_HISTORY_IMPL = os.environ.get("MEGATRON_DUAL_HISTORY_IMPL", "einsum").strip().lower()
_DUAL_CURRENT_KEY_IMPL = os.environ.get("MEGATRON_DUAL_CURRENT_KEY_IMPL", "cached").strip().lower()
_DUAL_KV_MODE_RAW = os.environ.get("MEGATRON_DUAL_KV_MODE", "cross").strip().lower().replace("-", "_")
_DUAL_KV_MODE_ALIASES = {
    "cross": "crossk_selfv",
    "cross_kv": "crossk_selfv",
    "crossk": "crossk_selfv",
    "cross_key": "crossk_selfv",
    "crossk_selfv": "crossk_selfv",
    "cross_key_self_value": "crossk_selfv",
    "self": "selfk_selfv",
    "self_kv": "selfk_selfv",
    "selfk": "selfk_selfv",
    "self_key": "selfk_selfv",
    "selfk_selfv": "selfk_selfv",
    "self_key_self_value": "selfk_selfv",
    "selfk_crossv": "selfk_crossv",
    "self_key_cross_value": "selfk_crossv",
}
_DUAL_KV_MODE = _DUAL_KV_MODE_ALIASES.get(_DUAL_KV_MODE_RAW, _DUAL_KV_MODE_RAW)
if _DUAL_KV_MODE not in {"crossk_selfv", "selfk_selfv", "selfk_crossv"}:
    raise ValueError("MEGATRON_DUAL_KV_MODE must be one of: cross, self, selfk_crossv")
_DUAL_CACHE_HISTORY_LOGITS = _env_bool("MEGATRON_DUAL_CACHE_HISTORY_LOGITS", False)
_DUAL_CACHE_HISTORY_PREFIX = _env_bool("MEGATRON_DUAL_CACHE_HISTORY_PREFIX", False)
_DUAL_CACHE_HISTORY_PREFIX_PAIR = _env_bool("MEGATRON_DUAL_CACHE_HISTORY_PREFIX_PAIR", False)
_DUAL_RECOMPUTE_HISTORY_VALUES = _env_bool("MEGATRON_DUAL_RECOMPUTE_HISTORY_VALUES", False)
_DUAL_RECOMPUTE_HISTORY_KEYS = _env_bool("MEGATRON_DUAL_RECOMPUTE_HISTORY_KEYS", False)
_DUAL_SCALAR_MIX_IMPL = os.environ.get("MEGATRON_DUAL_SCALAR_MIX_IMPL", "rmsnorm").strip().lower()
_DUAL_DEFER_STREAM_MIX = _env_bool("MEGATRON_DUAL_DEFER_STREAM_MIX", False)
_DUAL_MERGE_IMPL = os.environ.get("MEGATRON_DUAL_MERGE_IMPL", "stack").strip().lower()
_DUAL_TORCH_COMPILE = _env_bool("MEGATRON_DUAL_TORCH_COMPILE", False)
_DUAL_TORCH_COMPILE_DEPTH = _env_bool("MEGATRON_DUAL_TORCH_COMPILE_DEPTH", _DUAL_TORCH_COMPILE)
_DUAL_TORCH_COMPILE_MIX = _env_bool("MEGATRON_DUAL_TORCH_COMPILE_MIX", _DUAL_TORCH_COMPILE)
_DUAL_TORCH_COMPILE_MODE = os.environ.get("MEGATRON_DUAL_TORCH_COMPILE_MODE", "reduce-overhead")
_DUAL_Q_INIT = os.environ.get("MEGATRON_DUAL_Q_INIT", "zero_antisym").strip().lower()
_DUAL_Q_INIT_EPS = float(os.environ.get("MEGATRON_DUAL_Q_INIT_EPS", "1.0e-3"))
_DUAL_Q_BALANCED = _env_bool("MEGATRON_DUAL_Q_BALANCED", True)
_DUAL_AB_INIT = os.environ.get("MEGATRON_DUAL_AB_INIT", "neutral").strip().lower()
_DUAL_STATIC_ALPHA_LOGIT = os.environ.get("MEGATRON_DUAL_STATIC_ALPHA_LOGIT")
_DUAL_STATIC_BETA_LOGIT = os.environ.get("MEGATRON_DUAL_STATIC_BETA_LOGIT")
_DUAL_AB_SYMMETRY_BREAK_LOGIT = float(os.environ.get("MEGATRON_DUAL_AB_SYMMETRY_BREAK_LOGIT", "0.02"))
_DUAL_STATIC_RESIDUAL_LOGIT = os.environ.get("MEGATRON_DUAL_STATIC_RESIDUAL_LOGIT")
_DUAL_ATTRES_LIKE_FIRST_ATTN = _env_bool("MEGATRON_DUAL_ATTRES_LIKE_FIRST_ATTN", False)
_DUAL_ATTRES_LIMIT = _env_bool("MEGATRON_DUAL_ATTRES_LIMIT", False)
_DUAL_CURRENT_PARTIAL_RESIDUAL = _env_bool("MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL", True)
_DUAL_FUSE_PARTIAL_KEEP_MIX = _env_bool("MEGATRON_DUAL_FUSE_PARTIAL_KEEP_MIX", False)
_DUAL_TRITON_FUSED_RESIDUAL_READ = _env_bool("MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ", False)
_DUAL_TRITON_FUSED_RESIDUAL_READ_STRICT = _env_bool("MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ_STRICT", False)
_DUAL_TRITON_FUSED_RESIDUAL_READ_DEBUG = _env_bool("MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ_DEBUG", False)
_DUAL_TRITON_CURRENT_KEY_NORM = _env_bool("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM", False)
_DUAL_TRITON_HISTORY_PREFIX_PAIR = _env_bool("MEGATRON_DUAL_TRITON_HISTORY_PREFIX_PAIR", False)
_DUAL_TRITON_HISTORY_APPEND_PACK = _env_bool("MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK", False)
_DUAL_TRITON_HISTORY_APPEND_PACK_STRICT = _env_bool(
    "MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK_STRICT", False
)
_DUAL_TRITON_DEFERRED_MERGE = _env_bool("MEGATRON_DUAL_TRITON_DEFERRED_MERGE", False)
_DUAL_TRITON_FUSED_RESIDUAL_READ_DISABLED = False
_DUAL_TRITON_FUSED_RESIDUAL_READ_GATE_DEBUG_PRINTED = False
_DUAL_AB_SOURCE = os.environ.get("MEGATRON_DUAL_AB_SOURCE", "attended").strip().lower()
if _DUAL_AB_SOURCE not in {"attended", "partial"}:
    raise ValueError("MEGATRON_DUAL_AB_SOURCE must be one of: attended, partial")
_DUAL_KEEP_SOURCE = os.environ.get("MEGATRON_DUAL_KEEP_SOURCE", "attended").strip().lower()
if _DUAL_KEEP_SOURCE not in {"attended", "partial", "static"}:
    raise ValueError("MEGATRON_DUAL_KEEP_SOURCE must be one of: attended, partial, static")
_DUAL_EXCLUDE_RAW_HISTORY = _env_bool("MEGATRON_DUAL_EXCLUDE_RAW_HISTORY", False)
_DUAL_RAWLESS_RESIDUAL = _env_bool("MEGATRON_DUAL_RAWLESS_RESIDUAL", False)
_DUAL_EMBED_AS_PARTIAL = _env_bool("MEGATRON_DUAL_EMBED_AS_PARTIAL", False)
_DUAL_DEFER_PARTIAL_KEEP_MIX = (
    _DUAL_FUSE_PARTIAL_KEEP_MIX
    and _DUAL_KEEP_SOURCE == "partial"
    and _DUAL_CURRENT_PARTIAL_RESIDUAL
    and _DUAL_DEFER_STREAM_MIX
    and not _DUAL_ATTRES_LIMIT
)


def _optional_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _maybe_compile(fn, name: str, enabled: bool):
    if not enabled or not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, mode=_DUAL_TORCH_COMPILE_MODE, dynamic=False)
    except Exception as exc:
        print(f"[dual_compile] disabled {name}: {exc}", flush=True)
        return fn


def _mark_sequence_parallel_parameters(*params: nn.Parameter) -> None:
    for param in params:
        param.sequence_parallel = True


class FixedRMSNorm(nn.Module):
    def __init__(self, eps: float, *, force_fp32: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.force_fp32 = bool(force_fp32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(dtype=x.dtype)
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)


def _dual_history_norm_cache_key(history_key_norm: nn.Module) -> tuple:
    # FixedRMSNorm is fully described by these two public fields and has no
    # learnable state, so equal instances are safe common subexpressions.
    # Other modules may have behavior outside parameters/eps; keep them by
    # identity rather than guessing at equivalence.
    if type(history_key_norm) is FixedRMSNorm:
        return (
            "fixed_rms_norm",
            history_key_norm.eps,
            history_key_norm.force_fp32,
        )
    return ("module", id(history_key_norm))


def _rebuild_attres_history_keys(
    blocks: tuple[torch.Tensor, ...],
    norm_eps: float,
    force_norm_fp32: bool,
) -> torch.Tensor:
    packed = torch.stack(blocks, dim=0)
    if force_norm_fp32 and packed.is_cuda and triton_fixed_rms_norm is not None:
        return triton_fixed_rms_norm(
            packed,
            norm_eps,
            block_h=_env_int("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
            num_warps=_env_int("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
        )
    norm_input = packed.float() if force_norm_fp32 else packed
    keys = F.rms_norm(norm_input, (packed.shape[-1],), eps=norm_eps)
    return keys.to(dtype=packed.dtype) if force_norm_fp32 else keys


def _rebuild_dual_history_keys(
    blocks: tuple[torch.Tensor, ...],
    norm_eps: float,
    force_norm_fp32: bool,
) -> torch.Tensor:
    packed = torch.stack(blocks, dim=0)
    history, seq_len, batch2, hidden_size = packed.shape
    streams = packed.view(history, seq_len, batch2 // 2, 2, hidden_size)
    if force_norm_fp32 and streams.is_cuda and triton_dual_history_key_pack_norm is not None:
        return triton_dual_history_key_pack_norm(
            streams,
            norm_eps,
            cross_key=_DUAL_KV_MODE == "crossk_selfv",
            block_h=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
            num_warps=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
        )
    if force_norm_fp32 and streams.is_cuda and triton_fixed_rms_norm is not None:
        keys = triton_fixed_rms_norm(
            streams,
            norm_eps,
            block_h=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
            num_warps=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
        )
    else:
        norm_input = streams.float() if force_norm_fp32 else streams
        keys = F.rms_norm(norm_input, (hidden_size,), eps=norm_eps)
        if force_norm_fp32:
            keys = keys.to(dtype=streams.dtype)
    keys = keys.permute(3, 0, 2, 1, 4)
    if _DUAL_KV_MODE == "crossk_selfv":
        keys = keys.flip(0)
    return keys.contiguous()


class _AttresHistoryLogitsRecomputeKeys(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        keys: torch.Tensor,
        queries: torch.Tensor,
        norm_eps: float,
        force_norm_fp32: bool,
        *history_blocks: torch.Tensor,
    ) -> torch.Tensor:
        ctx.norm_eps = float(norm_eps)
        ctx.force_norm_fp32 = bool(force_norm_fp32)
        ctx.key_shape = tuple(keys.shape)
        ctx.save_for_backward(queries, *history_blocks)
        return torch.einsum("qh,nsbh->qnbs", queries, keys)

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        queries, *history_blocks = ctx.saved_tensors
        keys = _rebuild_attres_history_keys(
            tuple(history_blocks),
            ctx.norm_eps,
            ctx.force_norm_fp32,
        )
        history, seq_len, batch_size, hidden_size = ctx.key_shape
        flat_keys = keys.permute(0, 2, 1, 3).reshape(-1, hidden_size)
        flat_grad = grad_logits.permute(1, 2, 3, 0).reshape(-1, 2)
        grad_keys = torch.mm(flat_grad, queries).reshape(
            history, batch_size, seq_len, hidden_size
        ).permute(0, 2, 1, 3)
        grad_queries = torch.mm(flat_keys.transpose(0, 1), flat_grad).transpose(0, 1)
        return grad_keys, grad_queries, None, None, *(None for _ in history_blocks)


class _DualHistoryLogitsRecomputeKeys(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        keys: torch.Tensor,
        queries: torch.Tensor,
        norm_eps: float,
        force_norm_fp32: bool,
        *history_blocks: torch.Tensor,
    ) -> torch.Tensor:
        ctx.norm_eps = float(norm_eps)
        ctx.force_norm_fp32 = bool(force_norm_fp32)
        ctx.key_shape = tuple(keys.shape)
        ctx.save_for_backward(queries, *history_blocks)
        return torch.einsum("rdh,dnbsh->rdnbs", queries, keys)

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        queries, *history_blocks = ctx.saved_tensors
        keys = _rebuild_dual_history_keys(
            tuple(history_blocks),
            ctx.norm_eps,
            ctx.force_norm_fp32,
        )
        depth, history, batch_size, seq_len, hidden_size = ctx.key_shape
        flat_keys = keys.reshape(depth, -1, hidden_size)
        flat_grad = grad_logits.permute(1, 2, 3, 4, 0).reshape(depth, -1, 2)
        query_drh = queries.permute(1, 0, 2)
        grad_keys = torch.bmm(flat_grad, query_drh).reshape_as(keys)
        grad_queries = torch.bmm(flat_keys.transpose(1, 2), flat_grad).permute(2, 0, 1)
        return grad_keys, grad_queries, None, None, None, *(None for _ in history_blocks)


def _depth_logits(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    return torch.matmul(keys, query)


def _depth_weighted_sum(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if _ATTRES_TRITON_WEIGHTED_SUM and int(weights.shape[0]) > 1:
        if triton_attres_weighted_sum is None or not _attres_residual_triton_is_available():
            raise RuntimeError("MEGATRON_ATTRES_TRITON_WEIGHTED_SUM=1 requires Triton")
        kernel_weights = weights.to(dtype=values.dtype).contiguous()
        return triton_attres_weighted_sum(kernel_weights, values.contiguous())
    # Keep the contraction as one autograd node.  The addcmul loop is a little
    # cheaper in eager mode, but it can produce gradient dtype mismatches with
    # Megatron tensor-parallel reduce-scatter in backward.
    return torch.einsum("nsb,nsbd->sbd", weights, values)


def _attres_history_prefix_pair(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_query: torch.Tensor,
    mlp_query: torch.Tensor,
    *,
    history_blocks: Optional[list[torch.Tensor]] = None,
    history_norm_eps: float = 1.0e-5,
    history_norm_fp32: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    queries = torch.stack(
        (attn_query.to(dtype=keys.dtype), mlp_query.to(dtype=keys.dtype)),
        dim=0,
    )
    if _ATTRES_RECOMPUTE_HISTORY_KEYS and history_blocks is not None:
        logits = _AttresHistoryLogitsRecomputeKeys.apply(
            keys,
            queries,
            history_norm_eps,
            history_norm_fp32,
            *history_blocks,
        )
    else:
        logits = torch.einsum("qh,nsbh->qnbs", queries, keys)
    if _ATTRES_TRITON_HISTORY_PREFIX_PAIR:
        if triton_attres_history_prefix_pair is None or not _dual_residual_triton_is_available():
            raise RuntimeError("MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_PAIR=1 requires Triton")
        history_logit, history_attended = triton_attres_history_prefix_pair(
            logits.contiguous(),
            values.contiguous(),
            history_blocks=history_blocks if _ATTRES_RECOMPUTE_HISTORY_VALUES else None,
        )
    else:
        history_max = logits.max(dim=1).values
        history_scores = (logits - history_max.unsqueeze(1)).exp()
        history_denom = history_scores.sum(dim=1)
        history_logit = history_max + history_denom.log()
        history_num = torch.einsum(
            "qnbs,nsbh->qsbh",
            history_scores.to(dtype=values.dtype),
            values,
        )
        history_attended = history_num / history_denom.permute(0, 2, 1).unsqueeze(-1).to(
            dtype=history_num.dtype
        )
    history_logit = history_logit.permute(0, 2, 1)
    return (
        (history_logit[0], history_attended[0]),
        (history_logit[1], history_attended[1]),
    )


def _dual_history_logits(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if _DUAL_HISTORY_IMPL in {"bmm", "bmm_full"}:
        depth, history, batch, seq, hidden = keys.shape
        logits = torch.bmm(
            keys.reshape(depth, history * batch * seq, hidden),
            query.unsqueeze(-1),
        ).squeeze(-1)
        return logits.view(depth, history, batch, seq)
    if _DUAL_HISTORY_IMPL == "mulsum":
        return (keys * query.view(query.shape[0], 1, 1, 1, query.shape[1])).sum(dim=-1)
    return torch.einsum("dh,dnbsh->dnbs", query, keys)


def _dual_current_logits(current_key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if _DUAL_HISTORY_IMPL in {"bmm", "bmm_full"}:
        depth, batch, seq, hidden = current_key.shape
        logits = torch.bmm(
            current_key.reshape(depth, batch * seq, hidden),
            query.unsqueeze(-1),
        ).squeeze(-1)
        return logits.view(depth, batch, seq)
    if _DUAL_HISTORY_IMPL == "mulsum":
        return (current_key * query.view(query.shape[0], 1, 1, query.shape[1])).sum(dim=-1)
    return torch.einsum("dh,dbsh->dbs", query, current_key)


def _dual_current_logits_from_key_streams(
    current_key_streams: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    query = query.to(dtype=current_key_streams.dtype)
    if _DUAL_KV_MODE == "crossk_selfv":
        key0 = current_key_streams[:, :, 1, :]
        key1 = current_key_streams[:, :, 0, :]
    else:
        key0 = current_key_streams[:, :, 0, :]
        key1 = current_key_streams[:, :, 1, :]
    logit0 = torch.matmul(key0, query[0])
    logit1 = torch.matmul(key1, query[1])
    return torch.stack((logit0, logit1), dim=0).permute(0, 2, 1)


def _dual_history_weighted_sum(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if _DUAL_HISTORY_IMPL == "bmm_full":
        depth, history, batch, seq, hidden = values.shape
        weighted = torch.bmm(
            weights.permute(0, 2, 3, 1).reshape(depth * batch * seq, 1, history),
            values.permute(0, 2, 3, 1, 4).reshape(depth * batch * seq, history, hidden),
        ).squeeze(1)
        return weighted.view(depth, batch, seq, hidden).permute(2, 1, 0, 3)
    if _DUAL_HISTORY_IMPL == "mulsum":
        return (weights.unsqueeze(-1) * values).sum(dim=1).permute(2, 1, 0, 3)
    return torch.einsum("dnbs,dnbsh->sbdh", weights, values)


def _dual_history_prefix(
    keys: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _dual_history_logits(keys, query.to(dtype=keys.dtype))
    history_max = logits.max(dim=1).values
    history_scores = (logits - history_max.unsqueeze(1)).exp()
    history_denom = history_scores.sum(dim=1)
    history_num = torch.einsum("dnbs,dnbsh->sbdh", history_scores.to(dtype=values.dtype), values)
    history_attended = history_num / history_denom.permute(2, 1, 0).unsqueeze(-1).to(dtype=history_num.dtype)
    history_logit = history_max + history_denom.log()
    return history_logit, history_attended


def _dual_history_prefix_pair(
    keys: torch.Tensor,
    values: torch.Tensor,
    attn_query: torch.Tensor,
    mlp_query: torch.Tensor,
    *,
    history_blocks: Optional[list[torch.Tensor]] = None,
    history_norm_eps: float = 1.0e-5,
    history_norm_fp32: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    queries = torch.stack(
        (attn_query.to(dtype=keys.dtype), mlp_query.to(dtype=keys.dtype)),
        dim=0,
    )
    if _DUAL_RECOMPUTE_HISTORY_KEYS and history_blocks is not None:
        logits = _DualHistoryLogitsRecomputeKeys.apply(
            keys,
            queries,
            history_norm_eps,
            history_norm_fp32,
            *history_blocks,
        )
    else:
        logits = torch.einsum("rdh,dnbsh->rdnbs", queries, keys)
    if _DUAL_TRITON_HISTORY_PREFIX_PAIR:
        if triton_history_prefix_pair is None or not _dual_residual_triton_is_available():
            raise RuntimeError("MEGATRON_DUAL_TRITON_HISTORY_PREFIX_PAIR=1 requires Triton")
        history_logit, history_attended = triton_history_prefix_pair(
            logits.contiguous(),
            values.contiguous(),
            history_blocks=history_blocks if _DUAL_RECOMPUTE_HISTORY_VALUES else None,
            flip_values=_DUAL_KV_MODE == "selfk_crossv",
        )
    else:
        history_max = logits.max(dim=2).values
        history_scores = (logits - history_max.unsqueeze(2)).exp()
        history_denom = history_scores.sum(dim=2)
        history_logit = history_max + history_denom.log()
        history_num = torch.einsum("rdnbs,dnbsh->rsbdh", history_scores.to(dtype=values.dtype), values)
        history_attended = history_num / history_denom.permute(0, 3, 2, 1).unsqueeze(-1).to(dtype=history_num.dtype)
    return (
        (history_logit[0], history_attended[0]),
        (history_logit[1], history_attended[1]),
    )


def _dual_attend_from_history_prefix(
    history_prefix: tuple[torch.Tensor, torch.Tensor],
    current_logits: Optional[torch.Tensor],
    current_value_sbdh: Optional[torch.Tensor],
    output_dtype: Optional[torch.dtype],
    *,
    current_value_cross: bool = False,
) -> torch.Tensor:
    history_logit, history_attended = history_prefix
    if current_logits is None or current_value_sbdh is None:
        return history_attended if output_dtype is None else history_attended.to(dtype=output_dtype)

    history_weight_sbd = (history_logit - current_logits).sigmoid().permute(2, 1, 0)
    current_weight_sbd = (current_logits - history_logit).sigmoid().permute(2, 1, 0)
    history_part = history_attended * history_weight_sbd.unsqueeze(-1).to(dtype=history_attended.dtype)
    if current_value_cross:
        current_value_sbdh = current_value_sbdh.to(dtype=history_attended.dtype)
        current_weight_sbd = current_weight_sbd.to(dtype=history_attended.dtype)
        current_part = torch.stack(
            (
                current_value_sbdh[:, :, 1, :] * current_weight_sbd[:, :, 0:1],
                current_value_sbdh[:, :, 0, :] * current_weight_sbd[:, :, 1:2],
            ),
            dim=2,
        )
    else:
        current_part = current_value_sbdh.to(dtype=history_attended.dtype) * current_weight_sbd.unsqueeze(-1).to(
            dtype=history_attended.dtype
        )
    attended = history_part + current_part
    return attended if output_dtype is None else attended.to(dtype=output_dtype)


def _dual_depth_attention_eager(
    keys: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    *,
    current_key: Optional[torch.Tensor] = None,
    current_value: Optional[torch.Tensor] = None,
    current_streams: Optional[torch.Tensor] = None,
    current_key_streams: Optional[torch.Tensor] = None,
    history_logits: Optional[torch.Tensor] = None,
    history_prefix: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    output_dtype: Optional[torch.dtype] = None,
    current_value_cross: bool = False,
) -> torch.Tensor:
    query = query.to(dtype=keys.dtype)
    values = values.to(dtype=keys.dtype)
    has_current_streams = current_streams is not None
    has_current_key_streams = current_key_streams is not None
    if current_key is None and current_value is None and not has_current_streams and not has_current_key_streams and keys.shape[1] == 1:
        attended = values[:, 0].permute(2, 1, 0, 3).contiguous()
        # A one-entry history makes the softmax independent of q.  Keep an
        # exact zero dependency so Megatron's overlap-grad bucket still sees
        # this replicated parameter become ready on the first global branch.
        attended = attended + query.sum().to(dtype=attended.dtype) * 0.0
        return attended if output_dtype is None else attended.to(dtype=output_dtype)
    if history_prefix is not None:
        if has_current_key_streams:
            current_streams = current_streams.to(dtype=keys.dtype)
            current_key_streams = current_key_streams.to(dtype=keys.dtype)
            if keys.shape[1] == 0:
                if current_value_cross:
                    current_streams = torch.stack((current_streams[:, :, 1, :], current_streams[:, :, 0, :]), dim=2)
                return current_streams if output_dtype is None else current_streams.to(dtype=output_dtype)
            current_logits = _dual_current_logits_from_key_streams(current_key_streams, query)
            current_value_sbdh = current_streams
        elif current_key is not None and current_value is not None:
            current_key = current_key.to(dtype=keys.dtype)
            if keys.shape[1] == 0:
                attended = current_value.to(dtype=keys.dtype).permute(2, 1, 0, 3)
                return attended if output_dtype is None else attended.to(dtype=output_dtype)
            current_logits = _dual_current_logits(current_key, query)
            current_value_sbdh = current_value.to(dtype=keys.dtype).permute(2, 1, 0, 3)
        else:
            current_logits = None
            current_value_sbdh = None
        return _dual_attend_from_history_prefix(
            history_prefix,
            current_logits,
            current_value_sbdh,
            output_dtype,
            current_value_cross=current_value_cross and has_current_key_streams,
        )
    logits = history_logits if history_logits is not None else _dual_history_logits(keys, query)
    if current_key is None and current_value is None and not has_current_streams and not has_current_key_streams:
        weights = logits.softmax(dim=1).to(dtype=keys.dtype)
        attended = _dual_history_weighted_sum(weights, values)
        return attended if output_dtype is None else attended.to(dtype=output_dtype)

    if has_current_key_streams:
        current_streams = current_streams.to(dtype=keys.dtype)
        current_key_streams = current_key_streams.to(dtype=keys.dtype)
        if keys.shape[1] == 0:
            if current_value_cross:
                current_streams = torch.stack((current_streams[:, :, 1, :], current_streams[:, :, 0, :]), dim=2)
            return current_streams if output_dtype is None else current_streams.to(dtype=output_dtype)
        current_logits = _dual_current_logits_from_key_streams(current_key_streams, query)
    else:
        current_key = current_key.to(dtype=keys.dtype)
        if keys.shape[1] == 0:
            attended = current_value.to(dtype=keys.dtype).permute(2, 1, 0, 3)
            return attended if output_dtype is None else attended.to(dtype=output_dtype)
        current_logits = _dual_current_logits(current_key, query)
    if not _DUAL_SPLIT_CURRENT_SOFTMAX:
        weights = torch.cat((logits, current_logits.unsqueeze(1)), dim=1).softmax(dim=1).to(dtype=keys.dtype)
        history_count = values.shape[1]
        attended = _dual_history_weighted_sum(weights[:, :history_count], values)
        if has_current_streams:
            current_weights = weights[:, history_count].permute(2, 1, 0)
            if current_value_cross:
                current_attended = torch.stack(
                    (
                        current_streams[:, :, 1, :] * current_weights[:, :, 0:1],
                        current_streams[:, :, 0, :] * current_weights[:, :, 1:2],
                    ),
                    dim=2,
                )
            else:
                current_attended = current_streams * current_weights.unsqueeze(-1)
            attended = attended + current_attended
        else:
            current_attended = weights[:, history_count].unsqueeze(-1) * current_value.to(dtype=keys.dtype)
            attended = attended + current_attended.permute(2, 1, 0, 3)
        return attended if output_dtype is None else attended.to(dtype=output_dtype)
    max_logits = torch.maximum(logits.max(dim=1).values, current_logits)
    history_scores = (logits - max_logits.unsqueeze(1)).exp()
    current_scores = (current_logits - max_logits).exp()
    denom = history_scores.sum(dim=1) + current_scores
    history_weights = (history_scores / denom.unsqueeze(1)).to(dtype=keys.dtype)
    current_weights = (current_scores / denom).to(dtype=keys.dtype)
    attended = _dual_history_weighted_sum(history_weights, values)
    if has_current_streams:
        current_weights_sbd = current_weights.permute(2, 1, 0)
        if current_value_cross:
            current_attended = torch.stack(
                (
                    current_streams[:, :, 1, :] * current_weights_sbd[:, :, 0:1],
                    current_streams[:, :, 0, :] * current_weights_sbd[:, :, 1:2],
                ),
                dim=2,
            )
        else:
            current_attended = current_streams * current_weights_sbd.unsqueeze(-1)
        attended = attended + current_attended
    else:
        current_attended = current_weights.unsqueeze(-1) * current_value.to(dtype=keys.dtype)
        attended = attended + current_attended.permute(2, 1, 0, 3)
    return attended if output_dtype is None else attended.to(dtype=output_dtype)


_dual_depth_attention_impl = _maybe_compile(
    _dual_depth_attention_eager,
    "dual_depth_attention",
    _DUAL_TORCH_COMPILE_DEPTH,
)


def _dual_depth_attention(*args, **kwargs) -> torch.Tensor:
    return _dual_depth_attention_impl(*args, **kwargs)


def _dual_scalar_mix_eager(
    attended_streams: torch.Tensor,
    fused_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    static_residual_param: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    force_fp32_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stream0 = attended_streams[:, :, 0, :]
    stream1 = attended_streams[:, :, 1, :]
    flat = attended_streams.reshape(attended_streams.shape[0], attended_streams.shape[1], -1)
    post_norm_eps = float(norm_eps) / max(1, flat.shape[-1])
    if _DUAL_SCALAR_MIX_IMPL == "prescale" and not force_fp32_norm:
        compute_dtype = flat.dtype
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
        fused_dynamic = torch.matmul(flat, fused_weight.to(dtype=compute_dtype)) * inv_rms
    elif force_fp32_norm:
        normed = F.rms_norm(flat.float(), (flat.shape[-1],), eps=post_norm_eps).to(dtype=flat.dtype)
        compute_dtype = normed.dtype
        fused_dynamic = torch.matmul(normed, fused_weight.to(dtype=compute_dtype))
    else:
        normed = F.rms_norm(flat, (flat.shape[-1],), eps=post_norm_eps)
        compute_dtype = normed.dtype
        fused_dynamic = torch.matmul(normed, fused_weight.to(dtype=compute_dtype))
    dynamic_ab, residual_logit = fused_dynamic.split((2 * 2, 1), dim=-1)
    alpha, beta = dynamic_ab.split(2, dim=-1)
    static_alpha = static_alpha.to(dtype=compute_dtype)
    static_beta = static_beta.to(dtype=compute_dtype)
    static_residual_param = static_residual_param.to(dtype=compute_dtype)
    alpha = (alpha * pre_branch_scale.to(dtype=compute_dtype) + static_alpha.view(1, 1, 2)).sigmoid()
    beta = ((beta * h_post_scale.to(dtype=compute_dtype) + static_beta.view(1, 1, 2)).sigmoid() * 2.0)
    keep = (residual_logit.squeeze(-1) * residual_scale.to(dtype=compute_dtype) + static_residual_param).sigmoid()
    branch_input = alpha[..., :1] * stream0 + alpha[..., 1:] * stream1
    mixed0 = keep.unsqueeze(-1) * stream0 + (1.0 - keep).unsqueeze(-1) * stream1
    mixed1 = (1.0 - keep).unsqueeze(-1) * stream0 + keep.unsqueeze(-1) * stream1
    return (
        branch_input.to(dtype=attended_streams.dtype),
        mixed0.to(dtype=attended_streams.dtype),
        mixed1.to(dtype=attended_streams.dtype),
        beta.to(dtype=attended_streams.dtype),
    )


_dual_scalar_mix_impl = _maybe_compile(_dual_scalar_mix_eager, "dual_scalar_mix", _DUAL_TORCH_COMPILE_MIX)


def _dual_scalar_mix_defer_eager(
    attended_streams: torch.Tensor,
    fused_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    static_residual_param: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    force_fp32_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stream0 = attended_streams[:, :, 0, :]
    stream1 = attended_streams[:, :, 1, :]
    flat = attended_streams.reshape(attended_streams.shape[0], attended_streams.shape[1], -1)
    post_norm_eps = float(norm_eps) / max(1, flat.shape[-1])
    if _DUAL_SCALAR_MIX_IMPL == "prescale" and not force_fp32_norm:
        compute_dtype = flat.dtype
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
        fused_dynamic = torch.matmul(flat, fused_weight.to(dtype=compute_dtype)) * inv_rms
    elif force_fp32_norm:
        normed = F.rms_norm(flat.float(), (flat.shape[-1],), eps=post_norm_eps).to(dtype=flat.dtype)
        compute_dtype = normed.dtype
        fused_dynamic = torch.matmul(normed, fused_weight.to(dtype=compute_dtype))
    else:
        normed = F.rms_norm(flat, (flat.shape[-1],), eps=post_norm_eps)
        compute_dtype = normed.dtype
        fused_dynamic = torch.matmul(normed, fused_weight.to(dtype=compute_dtype))
    dynamic_ab, residual_logit = fused_dynamic.split((2 * 2, 1), dim=-1)
    alpha, beta = dynamic_ab.split(2, dim=-1)
    static_alpha = static_alpha.to(dtype=compute_dtype)
    static_beta = static_beta.to(dtype=compute_dtype)
    static_residual_param = static_residual_param.to(dtype=compute_dtype)
    alpha = (alpha * pre_branch_scale.to(dtype=compute_dtype) + static_alpha.view(1, 1, 2)).sigmoid()
    beta = ((beta * h_post_scale.to(dtype=compute_dtype) + static_beta.view(1, 1, 2)).sigmoid() * 2.0)
    keep = (residual_logit.squeeze(-1) * residual_scale.to(dtype=compute_dtype) + static_residual_param).sigmoid()
    branch_input = alpha[..., :1] * stream0 + alpha[..., 1:] * stream1
    return (
        branch_input.to(dtype=attended_streams.dtype),
        stream0,
        stream1,
        beta.to(dtype=attended_streams.dtype),
        keep.to(dtype=attended_streams.dtype),
    )


_dual_scalar_mix_defer_impl = _maybe_compile(
    _dual_scalar_mix_defer_eager,
    "dual_scalar_mix_defer",
    _DUAL_TORCH_COMPILE_MIX,
)


def _dual_stream_keep_eager(
    source_streams: torch.Tensor,
    residual_weight: torch.Tensor,
    static_residual_param: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    force_fp32_norm: bool,
    static_only: bool,
) -> torch.Tensor:
    flat = source_streams.reshape(source_streams.shape[0], source_streams.shape[1], -1)
    if static_only:
        compute_dtype = torch.float32 if force_fp32_norm else flat.dtype
        zero = (
            residual_weight.to(dtype=compute_dtype).sum()
            + residual_scale.to(dtype=compute_dtype).sum()
        ) * 0.0
        keep = (static_residual_param.to(dtype=compute_dtype) + zero).sigmoid()
        return keep.expand(flat.shape[0], flat.shape[1]).to(dtype=source_streams.dtype)

    post_norm_eps = float(norm_eps) / max(1, flat.shape[-1])
    if _DUAL_SCALAR_MIX_IMPL == "prescale" and not force_fp32_norm:
        compute_dtype = flat.dtype
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
        residual_logit = torch.matmul(flat, residual_weight.to(dtype=compute_dtype)) * inv_rms
    elif force_fp32_norm:
        normed = F.rms_norm(flat.float(), (flat.shape[-1],), eps=post_norm_eps).to(dtype=flat.dtype)
        compute_dtype = normed.dtype
        residual_logit = torch.matmul(normed, residual_weight.to(dtype=compute_dtype))
    else:
        normed = F.rms_norm(flat, (flat.shape[-1],), eps=post_norm_eps)
        compute_dtype = normed.dtype
        residual_logit = torch.matmul(normed, residual_weight.to(dtype=compute_dtype))
    keep = (
        residual_logit.squeeze(-1) * residual_scale.to(dtype=compute_dtype)
        + static_residual_param.to(dtype=compute_dtype)
    ).sigmoid()
    return keep.to(dtype=source_streams.dtype)


_dual_stream_keep_impl = _maybe_compile(
    _dual_stream_keep_eager,
    "dual_stream_keep",
    _DUAL_TORCH_COMPILE_MIX,
)


def _dual_stream_keep_mix_current_eager(
    source_streams: torch.Tensor,
    residual_weight: torch.Tensor,
    static_residual_param: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    force_fp32_norm: bool,
    static_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = _dual_stream_keep_eager(
        source_streams,
        residual_weight,
        static_residual_param,
        residual_scale,
        norm_eps,
        force_fp32_norm,
        static_only,
    )
    keep_for_mix = keep.unsqueeze(-1).to(dtype=source_streams.dtype)
    stream0 = source_streams[:, :, 0, :]
    stream1 = source_streams[:, :, 1, :]
    if _DUAL_MERGE_IMPL == "lerp_addcmul":
        mixed0 = torch.lerp(stream1, stream0, keep_for_mix)
        mixed1 = torch.lerp(stream0, stream1, keep_for_mix)
    else:
        one_minus_keep = 1.0 - keep_for_mix
        mixed0 = keep_for_mix * stream0 + one_minus_keep * stream1
        mixed1 = one_minus_keep * stream0 + keep_for_mix * stream1
    return mixed0, mixed1, keep


_dual_stream_keep_mix_current_impl = _maybe_compile(
    _dual_stream_keep_mix_current_eager,
    "dual_stream_keep_mix_current",
    _DUAL_TORCH_COMPILE_MIX,
)


def _dual_stream_alpha_beta_eager(
    source_streams: torch.Tensor,
    dynamic_ab_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    norm_eps: float,
    force_fp32_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = source_streams.reshape(source_streams.shape[0], source_streams.shape[1], -1)
    post_norm_eps = float(norm_eps) / max(1, flat.shape[-1])
    if _DUAL_SCALAR_MIX_IMPL == "prescale" and not force_fp32_norm:
        compute_dtype = flat.dtype
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
        dynamic_ab = torch.matmul(flat, dynamic_ab_weight.to(dtype=compute_dtype)) * inv_rms
    elif force_fp32_norm:
        normed = F.rms_norm(flat.float(), (flat.shape[-1],), eps=post_norm_eps).to(dtype=flat.dtype)
        compute_dtype = normed.dtype
        dynamic_ab = torch.matmul(normed, dynamic_ab_weight.to(dtype=compute_dtype))
    else:
        normed = F.rms_norm(flat, (flat.shape[-1],), eps=post_norm_eps)
        compute_dtype = normed.dtype
        dynamic_ab = torch.matmul(normed, dynamic_ab_weight.to(dtype=compute_dtype))
    alpha, beta = dynamic_ab.split(2, dim=-1)
    alpha = (
        alpha * pre_branch_scale.to(dtype=compute_dtype)
        + static_alpha.to(dtype=compute_dtype).view(1, 1, 2)
    ).sigmoid()
    beta = (
        (
            beta * h_post_scale.to(dtype=compute_dtype)
            + static_beta.to(dtype=compute_dtype).view(1, 1, 2)
        ).sigmoid()
        * 2.0
    )
    return alpha.to(dtype=source_streams.dtype), beta.to(dtype=source_streams.dtype)


_dual_stream_alpha_beta_impl = _maybe_compile(
    _dual_stream_alpha_beta_eager,
    "dual_stream_alpha_beta",
    _DUAL_TORCH_COMPILE_MIX,
)


def init_dual_stream_query_(query: torch.Tensor, *, preferred_stream: int, balanced: bool = False) -> None:
    if query.ndim != 2 or query.shape[0] != 2:
        raise ValueError(f"Expected [2, dim] dual query, got {tuple(query.shape)}")
    target_row_norm = 0.02
    with torch.no_grad():
        base = torch.randn(query.shape, device=query.device, dtype=torch.float32)
        stream0 = F.normalize(base[0], dim=0)
        stream1 = base[1] - torch.dot(base[1], stream0) * stream0
        if torch.linalg.vector_norm(stream1) < 1.0e-6:
            stream1 = torch.roll(stream0, shifts=1)
        stream1 = F.normalize(stream1, dim=0)
        if _DUAL_Q_INIT in {"zero_antisym", "zero", "attres_zero"}:
            initialized = torch.stack((stream1, -stream1), dim=0) * (target_row_norm * max(0.0, float(_DUAL_Q_INIT_EPS)))
        elif _DUAL_Q_INIT in {"attres_like", "shared_antisym", "shared"}:
            eps = max(0.0, float(_DUAL_Q_INIT_EPS))
            q0 = F.normalize(stream0 + eps * stream1, dim=0)
            q1 = F.normalize(stream0 - eps * stream1, dim=0)
            initialized = torch.stack((q0, q1), dim=0) * target_row_norm
        else:
            initialized = torch.stack((stream0, stream1), dim=0) * target_row_norm
        if not balanced and _DUAL_Q_INIT not in {"zero_antisym", "zero", "attres_zero", "attres_like", "shared_antisym", "shared"}:
            gains = torch.tensor([0.75, 0.75], device=query.device, dtype=torch.float32)
            gains[int(preferred_stream) % 2] = 1.25
            initialized = initialized * gains.unsqueeze(-1)
        query.copy_(initialized.to(dtype=query.dtype))


def _train_param_dtype(config) -> torch.dtype:
    train_cfg = getattr(config, "train", None)
    if bool(getattr(train_cfg, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(train_cfg, "fp16", False)):
        return torch.float16
    return torch.float32


def _residual_attention_delta(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask=None,
    rotary_pos_emb=None,
) -> torch.Tensor:
    if not _RESIDUAL_TE_ATTENTION_EPILOGUE:
        return layer.attention_delta(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
    if triton_te_attention_delta is None or not _dual_residual_triton_is_available():
        raise RuntimeError("MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE=1 requires Triton")
    raw_attention = layer.attention_raw_output(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
    )
    return triton_te_attention_delta(hidden_states, raw_attention)


class AttentionResidualBlock(nn.Module):
    _history_key = "history"
    _cache_root_key = "attention_residual_cache"
    _history_includes_input_key = "history_includes_current_input"

    def __init__(self, config, layers: list[nn.Module]) -> None:
        super().__init__()
        self.full_history = str(config.model.variant).lower() == "attention_residual"
        if self.full_history and len(layers) != 1:
            raise ValueError("attention_residual full mode requires one Transformer layer per residual block")
        self.layers = nn.ModuleList(layers)
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.attn_res_norm = FixedRMSNorm(config.model.layernorm_epsilon)
        self.mlp_res_norm = FixedRMSNorm(config.model.layernorm_epsilon)
        param_dtype = _train_param_dtype(config)
        self.attn_qs = nn.ParameterList(
            nn.Parameter(torch.zeros(config.model.hidden_size, dtype=param_dtype))
            for _ in self.layers
        )
        self.mlp_qs = nn.ParameterList(
            nn.Parameter(torch.zeros(config.model.hidden_size, dtype=param_dtype))
            for _ in self.layers
        )
        self.use_separate_k_proj = bool(config.model.hyper_conn_attres_separate_k_proj)
        if self.use_separate_k_proj:
            self.attn_res_k_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size, bias=False)
            self.mlp_res_k_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size, bias=False)
            with torch.no_grad():
                self.attn_res_k_proj.weight.copy_(torch.eye(config.model.hidden_size))
                self.mlp_res_k_proj.weight.copy_(torch.eye(config.model.hidden_size))
        self.share_attn_mlp_history_keys = (
            not self.use_separate_k_proj
            and type(self.attn_res_norm) is type(self.mlp_res_norm)
            and not list(self.attn_res_norm.parameters())
            and not list(self.mlp_res_norm.parameters())
        )
        sequence_parallel_params = [*self.attn_qs, *self.mlp_qs]
        if self.use_separate_k_proj:
            sequence_parallel_params.extend(
                [
                    self.attn_res_k_proj.weight,
                    self.mlp_res_k_proj.weight,
                ]
            )
        _mark_sequence_parallel_parameters(*sequence_parallel_params)
        block_layers = max(1, int(config.model.hyper_conn_block_layers))
        self.max_history_len = (2 * config.model.num_layers) + 1 if self.full_history else (config.model.num_layers // block_layers) + 1
    def _get_blocks(self, partial_block: torch.Tensor, state: dict) -> list[torch.Tensor]:
        blocks = state.get(self._history_key)
        if blocks is None:
            blocks = [partial_block]
            state[self._history_key] = blocks
        elif isinstance(blocks, torch.Tensor):
            # Pipeline stages may hand us a single tensor rather than the
            # Python list expected by the block logic.  Treat it as one
            # completed history block instead of iterating over its seq dim.
            blocks = [blocks]
            state[self._history_key] = blocks
        elif isinstance(blocks, tuple):
            blocks = list(blocks)
            state[self._history_key] = blocks
        return blocks

    def _make_key(
        self,
        value: torch.Tensor,
        norm: nn.Module,
        k_proj: Optional[nn.Linear],
        *,
        current: bool = False,
    ) -> torch.Tensor:
        if current and _ATTRES_TRITON_CURRENT_KEY_NORM and isinstance(norm, FixedRMSNorm) and norm.force_fp32:
            if triton_fixed_rms_norm is None or not _dual_residual_triton_is_available():
                raise RuntimeError("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM=1 requires Triton")
            key = triton_fixed_rms_norm(
                value,
                norm.eps,
                block_h=_env_int("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
                num_warps=_env_int("MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
            )
        else:
            key = norm(value)
        if k_proj is not None:
            key = k_proj(key)
        return key

    @staticmethod
    def _append_packed(packed: Optional[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(0)
        return value if packed is None else torch.cat((packed, value), dim=0)

    def _append_cache_block(self, cache: dict, block: torch.Tensor) -> None:
        if len(cache["blocks"]) >= self.max_history_len:
            raise ValueError("attention residual block history exceeded max_history_len")
        cache["blocks"].append(block)
        if (
            _ATTRES_TRITON_HISTORY_APPEND_PACK
            and not self.share_attn_mlp_history_keys
            and _ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT
        ):
            raise ValueError("strict AttnRes history append/pack requires the shared history-key path")
        if _ATTRES_TRITON_HISTORY_APPEND_PACK and self.share_attn_mlp_history_keys:
            try:
                if triton_attres_history_append_pack is None or not _attres_residual_triton_is_available():
                    raise RuntimeError("MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK=1 requires Triton")
                if cache["attn_keys"] is None:
                    previous_shape = (0, *block.shape)
                    previous_keys = torch.empty(previous_shape, device=block.device, dtype=block.dtype)
                    previous_values = torch.empty_like(previous_keys)
                else:
                    previous_keys = cache["attn_keys"]
                    previous_values = cache["values"]
                shared_keys, values = triton_attres_history_append_pack(
                    previous_keys,
                    previous_values,
                    block,
                    self.attn_res_norm.eps,
                )
                cache["values"] = values
                cache["attn_keys"] = shared_keys
                cache["mlp_keys"] = shared_keys
                return
            except (RuntimeError, ValueError):
                if _ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT:
                    raise
        attn_key = self._make_key(
            block,
            self.attn_res_norm,
            self.attn_res_k_proj if self.use_separate_k_proj else None,
        )
        if self.use_separate_k_proj:
            mlp_key = self._make_key(block, self.mlp_res_norm, self.mlp_res_k_proj)
        elif self.share_attn_mlp_history_keys:
            # FixedRMSNorm has no learnable parameters, so attn/mlp history
            # keys are numerically identical when no separate K projection is
            # requested.  Share the cached history key while still computing
            # the current partial key at its own pre-attn / post-attn point.
            mlp_key = attn_key
        else:
            mlp_key = self._make_key(block, self.mlp_res_norm, None)
        cache["values"] = self._append_packed(cache["values"], block)
        cache["attn_keys"] = self._append_packed(cache["attn_keys"], attn_key)
        cache["mlp_keys"] = self._append_packed(cache["mlp_keys"], mlp_key)

    def _get_cache(self, partial_block: torch.Tensor, state: dict) -> dict:
        cache_root = state.setdefault(self._cache_root_key, {})
        cache = cache_root.get(id(self))
        if cache is not None:
            state[self._history_key] = cache["blocks"]
            return cache
        blocks = self._get_blocks(partial_block, state)
        cache = {"blocks": [], "values": None, "attn_keys": None, "mlp_keys": None}
        for block in blocks:
            self._append_cache_block(cache, block)
        cache_root[id(self)] = cache
        state[self._history_key] = cache["blocks"]
        return cache

    def _block_attn_res(
        self,
        cache: dict,
        partial_block: torch.Tensor,
        query: torch.Tensor,
        *,
        key_name: str,
        norm: nn.Module,
        k_proj: Optional[nn.Linear] = None,
        include_partial: bool = True,
        cached_history_logits: Optional[torch.Tensor] = None,
        history_prefix: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        values = cache["values"]
        keys = cache[key_name]
        if values is None or keys is None:
            raise ValueError("attention residual block requires at least one cached history block")
        values = values.to(dtype=keys.dtype)
        query = query.to(dtype=keys.dtype)
        if keys.ndim == 4:
            if history_prefix is not None:
                history_logit, history_attended = history_prefix
                if not include_partial:
                    return history_attended.to(dtype=partial_block.dtype)
                if _ATTRES_TRITON_CURRENT_FINALIZE:
                    try:
                        if triton_attres_current_finalize is None or not _attres_residual_triton_is_available():
                            raise RuntimeError("MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE=1 requires Triton")
                        if k_proj is not None or not isinstance(norm, FixedRMSNorm) or not norm.force_fp32:
                            raise ValueError(
                                "AttnRes fused current finalize requires parameterless FP32 FixedRMSNorm"
                            )
                        partial_key = self._make_key(partial_block, norm, k_proj, current=True)
                        partial_key = partial_key.to(dtype=keys.dtype)
                        attended = triton_attres_current_finalize(
                            history_logit,
                            history_attended,
                            partial_block,
                            partial_key,
                            query,
                        )
                        return attended.to(dtype=partial_block.dtype)
                    except (RuntimeError, ValueError):
                        if _ATTRES_TRITON_CURRENT_FINALIZE_STRICT:
                            raise
                partial_key = self._make_key(partial_block, norm, k_proj, current=True)
                partial_key = partial_key.to(dtype=keys.dtype)
                current_logit = _depth_logits(partial_key, query)
                history_weight = (history_logit - current_logit).sigmoid().unsqueeze(-1)
                current_weight = (current_logit - history_logit).sigmoid().unsqueeze(-1)
                attended = (
                    history_attended * history_weight.to(dtype=history_attended.dtype)
                    + partial_block.to(dtype=history_attended.dtype)
                    * current_weight.to(dtype=history_attended.dtype)
                )
                return attended.to(dtype=partial_block.dtype)
            if include_partial:
                partial_key = self._make_key(partial_block, norm, k_proj, current=True)
            else:
                partial_key = None
            if cached_history_logits is None:
                logits = _depth_logits(keys, query)
            else:
                if reuse_attres_history_logits is None:
                    raise RuntimeError("AttnRes paired history-logit cache is unavailable")
                logits = reuse_attres_history_logits(keys, query, cached_history_logits)
            if partial_key is not None:
                partial_key = partial_key.to(dtype=keys.dtype)
                partial_logits = _depth_logits(partial_key, query).unsqueeze(0)
                logits = torch.cat((logits, partial_logits), dim=0)
            weights = logits.softmax(dim=0)
            history_count = values.shape[0]
            attended = _depth_weighted_sum(weights[:history_count], values)
            if partial_key is not None:
                attended = attended + weights[history_count].unsqueeze(-1) * partial_block.to(dtype=keys.dtype)
            return attended.to(dtype=partial_block.dtype)
        raise ValueError(
            "attention residual expected stacked history [history, seq, batch, hidden], "
            f"got {tuple(keys.shape)}; this usually means --sequence-parallel leaked into a residual wrapper"
        )

    def forward(self, x: torch.Tensor, *, attention_mask=None, rotary_pos_emb=None, state: Optional[dict] = None) -> torch.Tensor:
        if state is None:
            raise ValueError("AttentionResidualBlock requires a mutable state dict")
        t0 = _attres_profile_now()
        cache = self._get_cache(x, state)
        _attres_profile_record("cache", t0)
        partial_block = x
        aux_loss = None
        block_started = False
        history_includes_current_input = bool(state.pop(self._history_includes_input_key, False))
        history_logits_pairs: dict[int, torch.Tensor] = {}
        history_prefix_pairs: dict[
            int,
            tuple[
                Optional[tuple[torch.Tensor, torch.Tensor]],
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ],
        ] = {}

        def get_history_prefix(
            layer_index: int,
            name: str,
        ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
            if not _ATTRES_CACHE_HISTORY_PREFIX_PAIR:
                return None
            if not self.share_attn_mlp_history_keys:
                raise ValueError("AttnRes paired history-prefix cache requires shared attention/MLP keys")
            if _ATTRES_RECOMPUTE_HISTORY_KEYS and self.use_separate_k_proj:
                raise ValueError("AttnRes history-key recompute does not support separate K projections")
            keys = cache["attn_keys"]
            values = cache["values"]
            if keys is None or values is None:
                return None
            if layer_index not in history_prefix_pairs:
                t0 = _attres_profile_now()
                history_prefix_pairs[layer_index] = _attres_history_prefix_pair(
                    keys,
                    values.to(dtype=keys.dtype),
                    self.attn_qs[layer_index],
                    self.mlp_qs[layer_index],
                    history_blocks=(
                        cache["blocks"]
                        if _ATTRES_RECOMPUTE_HISTORY_VALUES or _ATTRES_RECOMPUTE_HISTORY_KEYS
                        else None
                    ),
                    history_norm_eps=self.attn_res_norm.eps,
                    history_norm_fp32=self.attn_res_norm.force_fp32,
                )
                _attres_profile_record("history_prefix_pair_precompute", t0)
            return history_prefix_pairs[layer_index][0 if name == "attn" else 1]

        def get_cached_history_logits(layer_index: int, name: str) -> Optional[torch.Tensor]:
            if (
                _ATTRES_CACHE_HISTORY_PREFIX_PAIR
                or not _ATTRES_CACHE_HISTORY_LOGITS_PAIR
                or not self.share_attn_mlp_history_keys
            ):
                return None
            keys = cache["attn_keys"]
            if keys is None:
                return None
            if paired_attres_history_logits is None:
                raise RuntimeError("MEGATRON_ATTRES_CACHE_HISTORY_LOGITS_PAIR=1 requires paired-logit support")
            history_logits_pair = history_logits_pairs.get(layer_index)
            if history_logits_pair is None or int(history_logits_pair.shape[1]) != int(keys.shape[0]):
                queries = torch.stack(
                    (
                        self.attn_qs[layer_index].to(dtype=keys.dtype),
                        self.mlp_qs[layer_index].to(dtype=keys.dtype),
                    ),
                    dim=0,
                )
                history_logits_pair = paired_attres_history_logits(keys, queries)
                history_logits_pairs[layer_index] = history_logits_pair
            return history_logits_pair[0 if name == "attn" else 1]

        for layer_index, layer in enumerate(self.layers):
            current_is_already_in_history = (cache["blocks"][-1] is partial_block) or (not block_started and history_includes_current_input)
            if _ATTRES_CACHE_HISTORY_PREFIX_PAIR and not block_started and not current_is_already_in_history:
                t0 = _attres_profile_now()
                self._append_cache_block(cache, partial_block)
                history_logits_pairs.clear()
                history_prefix_pairs.clear()
                current_is_already_in_history = True
                _attres_profile_record("append_history", t0)
            include_current_partial = not (not block_started and current_is_already_in_history)
            t0 = _attres_profile_now()
            attn_hidden = self._block_attn_res(
                cache,
                partial_block,
                self.attn_qs[layer_index],
                key_name="attn_keys",
                norm=self.attn_res_norm,
                k_proj=self.attn_res_k_proj if self.use_separate_k_proj else None,
                include_partial=include_current_partial,
                cached_history_logits=get_cached_history_logits(layer_index, "attn"),
                history_prefix=get_history_prefix(layer_index, "attn"),
            )
            _attres_profile_record("attn_residual_read", t0)
            if not block_started:
                if not current_is_already_in_history:
                    t0 = _attres_profile_now()
                    self._append_cache_block(cache, partial_block)
                    history_logits_pairs.clear()
                    history_prefix_pairs.clear()
                    _attres_profile_record("append_history", t0)
                block_started = True
                history_includes_current_input = False
                t0 = _attres_profile_now()
                partial_block = _residual_attention_delta(
                    layer,
                    attn_hidden,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )
                _attres_profile_record("attention_delta", t0)
            else:
                t0 = _attres_profile_now()
                partial_block = partial_block + _residual_attention_delta(
                    layer,
                    attn_hidden,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )
                _attres_profile_record("attention_delta", t0)

            t0 = _attres_profile_now()
            mlp_hidden = self._block_attn_res(
                cache,
                partial_block,
                self.mlp_qs[layer_index],
                key_name="mlp_keys",
                norm=self.mlp_res_norm,
                k_proj=self.mlp_res_k_proj if self.use_separate_k_proj else None,
                cached_history_logits=get_cached_history_logits(layer_index, "mlp"),
                history_prefix=get_history_prefix(layer_index, "mlp"),
            )
            _attres_profile_record("mlp_residual_read", t0)
            if self.full_history:
                self._append_cache_block(cache, partial_block)
                history_logits_pairs.clear()
                history_prefix_pairs.clear()
            t0 = _attres_profile_now()
            partial_block = partial_block + layer.mlp_delta(mlp_hidden)
            _attres_profile_record("mlp_delta", t0)
            if self.full_history:
                self._append_cache_block(cache, partial_block)
                history_logits_pairs.clear()
                history_prefix_pairs.clear()
            if layer.last_aux_loss is not None:
                aux_loss = layer.last_aux_loss if aux_loss is None else aux_loss + layer.last_aux_loss
        state[self._history_key] = cache["blocks"]
        self.last_aux_loss = aux_loss
        return partial_block


class AttentionResidualOutputAggregation(nn.Module):
    _history_key = "history"

    def __init__(self, config) -> None:
        super().__init__()
        self.norm = FixedRMSNorm(config.model.layernorm_epsilon)
        self.proj = nn.Linear(config.model.hidden_size, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        _mark_sequence_parallel_parameters(self.proj.weight)

    @staticmethod
    def _final_values(blocks: list[torch.Tensor], hidden_states: torch.Tensor) -> list[torch.Tensor]:
        # blocks[0] is the raw embedding/input history.  The original
        # attention_residual_block aggregates completed block outputs only.
        return [*blocks[1:], hidden_states]

    def forward(self, hidden_states: torch.Tensor, state: dict) -> torch.Tensor:
        blocks = state.get(self._history_key)
        if isinstance(blocks, torch.Tensor):
            blocks = [blocks]
        if not blocks:
            raise ValueError("attention residual output aggregation requires block history")
        t0 = _attres_profile_now()
        values = torch.stack(self._final_values(blocks, hidden_states), dim=0)
        keys = self.norm(values)
        values = values.to(dtype=keys.dtype)
        query = self.proj.weight.squeeze(0).to(dtype=keys.dtype)
        if keys.ndim == 4:
            logits = _depth_logits(keys, query)
            weights = logits.softmax(dim=0)
            out = _depth_weighted_sum(weights, values)
            _attres_profile_record("output_aggregation", t0)
            return out.to(dtype=hidden_states.dtype)
        raise ValueError(
            "attention residual output aggregation expected [history, seq, batch, hidden], "
            f"got {tuple(keys.shape)}; this usually means --sequence-parallel leaked into a residual wrapper"
        )


class _DualResidualSublayerState(nn.Module):
    """All learnable and normalization state for one atomic residual read."""

    def __init__(
        self,
        config,
        *,
        global_layer_index: int,
        dual_norm_fp32: bool,
        param_dtype: torch.dtype,
        static_residual_logit: float,
    ) -> None:
        super().__init__()
        num_streams = 2
        hidden_size = int(config.model.hidden_size)
        init_stream = global_layer_index % num_streams

        self.history_key_norm = FixedRMSNorm(
            config.model.layernorm_epsilon,
            force_fp32=dual_norm_fp32,
        )
        self.norm = FixedRMSNorm(
            config.model.layernorm_epsilon,
            force_fp32=dual_norm_fp32,
        )
        self.pre_branch_scale = nn.Parameter(torch.full((), 1e-2, dtype=param_dtype))
        self.h_post_scale = nn.Parameter(torch.full((), 1e-2, dtype=param_dtype))
        self.residual_scale = nn.Parameter(torch.full((), 1e-2, dtype=param_dtype))

        if _DUAL_AB_INIT in {"near_attres", "attres_near", "near_attres_asym"}:
            alpha_base = _optional_float(_DUAL_STATIC_ALPHA_LOGIT, 0.0)
            alpha_eps = abs(float(_DUAL_AB_SYMMETRY_BREAK_LOGIT))
            static_alpha = torch.full((num_streams, 1), alpha_base, dtype=param_dtype)
            static_alpha[init_stream, 0] = alpha_base + alpha_eps
            static_alpha[1 - init_stream, 0] = alpha_base - alpha_eps
        elif _DUAL_AB_INIT in {"neutral", "zero", "balanced"}:
            static_alpha = torch.zeros((num_streams, 1), dtype=param_dtype)
            if _DUAL_STATIC_ALPHA_LOGIT is not None and _DUAL_STATIC_ALPHA_LOGIT != "":
                static_alpha.fill_(float(_DUAL_STATIC_ALPHA_LOGIT))
        else:
            alpha_logit = abs(_optional_float(_DUAL_STATIC_ALPHA_LOGIT, 1.0))
            static_alpha = torch.full((num_streams, 1), -alpha_logit, dtype=param_dtype)
            static_alpha[init_stream, 0] = alpha_logit
        self.static_alpha = nn.Parameter(static_alpha)
        self.dynamic_ab_fn = nn.Parameter(
            torch.zeros(hidden_size * num_streams, num_streams * 2, dtype=param_dtype)
        )

        if _DUAL_AB_INIT in {"near_attres", "attres_near", "near_attres_asym"}:
            beta_base = _optional_float(_DUAL_STATIC_BETA_LOGIT, 0.0)
            beta_eps = abs(float(_DUAL_AB_SYMMETRY_BREAK_LOGIT))
            static_beta = torch.full((num_streams,), beta_base, dtype=param_dtype)
            static_beta[init_stream] = beta_base + beta_eps
            static_beta[1 - init_stream] = beta_base - beta_eps
        elif _DUAL_AB_INIT in {"neutral", "zero", "balanced"}:
            static_beta = torch.zeros((num_streams,), dtype=param_dtype)
            if _DUAL_STATIC_BETA_LOGIT is not None and _DUAL_STATIC_BETA_LOGIT != "":
                static_beta.fill_(float(_DUAL_STATIC_BETA_LOGIT))
        else:
            beta_logit = abs(_optional_float(_DUAL_STATIC_BETA_LOGIT, 1.0))
            static_beta = torch.full((num_streams,), -beta_logit, dtype=param_dtype)
            static_beta[init_stream] = beta_logit
        self.static_beta = nn.Parameter(static_beta)
        self.static_residual_logit = nn.Parameter(
            torch.tensor(static_residual_logit, dtype=param_dtype)
        )
        self.dynamic_residual_mix_fn = nn.Parameter(
            torch.zeros(hidden_size * num_streams, 1, dtype=param_dtype)
        )
        self.q = nn.Parameter(torch.empty(num_streams, hidden_size, dtype=param_dtype))
        init_dual_stream_query_(
            self.q,
            preferred_stream=init_stream,
            balanced=_DUAL_Q_BALANCED,
        )

    def fused_weight(self) -> torch.Tensor:
        return torch.cat((self.dynamic_ab_fn, self.dynamic_residual_mix_fn), dim=-1)


class DualAttentionResidualBlock(nn.Module):
    _history_key = "history"
    _cache_key = "history_cache"
    _history_includes_input_key = "history_includes_current_input"
    _embed_as_partial_consumed_key = "dual_embed_as_partial_consumed"

    def __init__(self, config, layers: list[nn.Module], layer_index: int) -> None:
        super().__init__()
        self.full_history = str(config.model.variant).lower() == "dual_attention_residual"
        if self.full_history and len(layers) != 1:
            raise ValueError("dual_attention_residual full mode requires one Transformer layer per residual block")
        self.layers = nn.ModuleList(layers)
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.num_streams = 2
        dual_norm_fp32 = _env_bool("MEGATRON_DUAL_RMSNORM_FP32", True)
        param_dtype = _train_param_dtype(config)
        if _DUAL_STATIC_RESIDUAL_LOGIT is not None and _DUAL_STATIC_RESIDUAL_LOGIT != "":
            static_residual_logit = float(_DUAL_STATIC_RESIDUAL_LOGIT)
        else:
            default_static_residual_logit = 4.0 if str(getattr(config.model, "dual_kv_mode", "cross")).lower() in {"cross", "cross_kv"} else 8.0
            static_residual_logit = float(
                getattr(config.model, "hyper_conn_dual_static_residual_logit", default_static_residual_logit)
            )
        self.attn_states = nn.ModuleList()
        self.mlp_states = nn.ModuleList()
        for local_layer_index in range(len(self.layers)):
            state_kwargs = {
                "global_layer_index": layer_index + local_layer_index,
                "dual_norm_fp32": dual_norm_fp32,
                "param_dtype": param_dtype,
                "static_residual_logit": static_residual_logit,
            }
            self.attn_states.append(_DualResidualSublayerState(config, **state_kwargs))
            self.mlp_states.append(_DualResidualSublayerState(config, **state_kwargs))
        _mark_sequence_parallel_parameters(
            *(parameter for state in (*self.attn_states, *self.mlp_states) for parameter in state.parameters())
        )
        block_layers = max(1, int(config.model.hyper_conn_block_layers))
        self.max_history_len = (2 * config.model.num_layers) + 1 if self.full_history else (config.model.num_layers // block_layers) + 1

    def _to_stream_tensor(self, residuals: torch.Tensor) -> torch.Tensor:
        seq, batch2, hidden = residuals.shape
        batch = batch2 // self.num_streams
        return residuals.view(seq, batch, self.num_streams, hidden)

    def _from_stream_tensor(self, residuals: torch.Tensor) -> torch.Tensor:
        seq, batch, streams, hidden = residuals.shape
        return residuals.contiguous().view(seq, batch * streams, hidden)

    def _history_value(self, value: torch.Tensor) -> torch.Tensor:
        value = value.transpose(0, 2)
        if _DUAL_KV_MODE == "selfk_crossv":
            value = value.flip(0)
        return value.contiguous()

    def _normalize_history_key(
        self,
        value: torch.Tensor,
        history_key_norm: nn.Module,
    ) -> torch.Tensor:
        key = history_key_norm(value).transpose(0, 2)
        if _DUAL_KV_MODE == "crossk_selfv":
            key = key.flip(0)
        return key.contiguous()

    @staticmethod
    def _history_norm_cache_key(history_key_norm: nn.Module) -> tuple:
        return _dual_history_norm_cache_key(history_key_norm)

    def _history_keys(
        self,
        cache: dict,
        history_key_norm: nn.Module,
    ) -> Optional[torch.Tensor]:
        cache_key = self._history_norm_cache_key(history_key_norm)
        entry = cache["keys_by_norm"].get(cache_key)
        if entry is None:
            entry = {"norm": history_key_norm, "keys": None}
            cache["keys_by_norm"][cache_key] = entry
        if entry["keys"] is None and cache["blocks"]:
            packed_keys = None
            for block in cache["blocks"]:
                stream_value = self._to_stream_tensor(block)
                key = self._normalize_history_key(stream_value, history_key_norm)
                packed_keys = self._append_packed_history(packed_keys, key)
            entry["keys"] = packed_keys
        return entry["keys"]

    @staticmethod
    def _current_value_streams(value: torch.Tensor) -> torch.Tensor:
        if _DUAL_KV_MODE == "selfk_crossv":
            return value.flip(2).contiguous()
        return value

    @staticmethod
    def _append_packed_history(packed: Optional[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(1)
        return value if packed is None else torch.cat((packed, value), dim=1)

    def _get_cache(self, partial_block: torch.Tensor, state: dict) -> dict:
        cache = state.get(self._cache_key)
        if cache is not None:
            state[self._history_key] = cache["blocks"]
            return cache
        blocks = state.get(self._history_key)
        if blocks is None:
            blocks = [] if _DUAL_EMBED_AS_PARTIAL else [partial_block]
        elif isinstance(blocks, torch.Tensor):
            blocks = [blocks]
        elif isinstance(blocks, tuple):
            blocks = list(blocks)
        state[self._history_key] = blocks
        cache = {
            "blocks": [],
            "keys_by_norm": {},
            "values": None,
        }
        for residual_state in (*self.attn_states, *self.mlp_states):
            history_key_norm = residual_state.history_key_norm
            cache_key = self._history_norm_cache_key(history_key_norm)
            cache["keys_by_norm"].setdefault(
                cache_key,
                {"norm": history_key_norm, "keys": None},
            )
        for block in blocks:
            self._append_block(cache, block)
        state[self._cache_key] = cache
        state[self._history_key] = cache["blocks"]
        return cache

    def _append_block(self, cache: dict, partial_block: torch.Tensor) -> None:
        if len(cache["blocks"]) >= self.max_history_len:
            raise ValueError("dual attention residual block history exceeded max_history_len")
        stream_value = self._to_stream_tensor(partial_block)
        cache["blocks"].append(partial_block)
        previous_values = cache["values"]
        new_value = self._history_value(stream_value)
        next_values = self._append_packed_history(previous_values, new_value)
        for entry in cache["keys_by_norm"].values():
            history_key_norm = entry["norm"]
            if _DUAL_TRITON_HISTORY_APPEND_PACK and isinstance(history_key_norm, FixedRMSNorm):
                try:
                    if triton_dual_history_append_pack is None or not _dual_residual_triton_is_available():
                        raise RuntimeError("MEGATRON_DUAL_TRITON_HISTORY_APPEND_PACK=1 requires Triton")
                    if entry["keys"] is None:
                        seq_len, batch_size, _, hidden_size = stream_value.shape
                        previous_shape = (2, 0, batch_size, seq_len, hidden_size)
                        previous_keys = torch.empty(
                            previous_shape,
                            device=stream_value.device,
                            dtype=stream_value.dtype,
                        )
                        entry_previous_values = torch.empty_like(previous_keys)
                    else:
                        previous_keys = entry["keys"]
                        entry_previous_values = previous_values
                    entry["keys"], packed_values = triton_dual_history_append_pack(
                        previous_keys,
                        entry_previous_values,
                        stream_value,
                        history_key_norm.eps,
                        _DUAL_KV_MODE == "crossk_selfv",
                        _DUAL_KV_MODE == "selfk_crossv",
                    )
                    next_values = packed_values
                    continue
                except (RuntimeError, ValueError):
                    if _DUAL_TRITON_HISTORY_APPEND_PACK_STRICT:
                        raise
            key = self._normalize_history_key(stream_value, history_key_norm)
            entry["keys"] = self._append_packed_history(entry["keys"], key)
        cache["values"] = next_values

    def _dual_history_attention(
        self,
        cache: dict,
        partial_block: torch.Tensor,
        query: torch.Tensor,
        history_key_norm: nn.Module,
        *,
        include_current: bool,
        history_logits: Optional[torch.Tensor] = None,
        history_prefix: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        keys = (
            self._history_keys(cache, history_key_norm)
            if "keys_by_norm" in cache
            else cache.get("keys")
        )
        values = cache["values"]
        if keys is None or values is None:
            if include_current:
                return self._current_value_streams(self._to_stream_tensor(partial_block))
            raise ValueError("dual attention residual block requires at least one cached history block")
        current_value_cross = False
        if include_current:
            t0 = _dual_profile_now()
            current = self._to_stream_tensor(partial_block)
            if _DUAL_CURRENT_KEY_IMPL == "stream_norm":
                current_key = None
                current_value = None
                current_streams = current
                current_value_cross = _DUAL_KV_MODE == "selfk_crossv"
                current_key_streams = history_key_norm(current)
            else:
                current_key = self._normalize_history_key(current, history_key_norm)
                current_value = self._history_value(current)
                current_streams = None
                current_key_streams = None
            _dual_profile_record("current_prepare", t0)
        else:
            current_key = None
            current_value = None
            current_streams = None
            current_key_streams = None
        t0 = _dual_profile_now()
        attended = _dual_depth_attention(
            keys,
            values,
            query,
            current_key=current_key,
            current_value=current_value,
            current_streams=current_streams,
            current_key_streams=current_key_streams,
            history_logits=history_logits,
            history_prefix=history_prefix,
            output_dtype=partial_block.dtype,
            current_value_cross=current_value_cross,
        )
        _dual_profile_record("depth_attention", t0)
        return attended

    def _mix_from_attended(
        self,
        attended_streams: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
    ) -> tuple[torch.Tensor, ...]:
        if _DUAL_DEFER_STREAM_MIX:
            return _dual_scalar_mix_defer_impl(
                attended_streams,
                fused_weight,
                residual_state.static_alpha,
                residual_state.static_beta,
                residual_state.static_residual_logit,
                residual_state.pre_branch_scale,
                residual_state.h_post_scale,
                residual_state.residual_scale,
                residual_state.norm.eps,
                residual_state.norm.force_fp32,
            )
        return _dual_scalar_mix_impl(
            attended_streams,
            fused_weight,
            residual_state.static_alpha,
            residual_state.static_beta,
            residual_state.static_residual_logit,
            residual_state.pre_branch_scale,
            residual_state.h_post_scale,
            residual_state.residual_scale,
            residual_state.norm.eps,
            residual_state.norm.force_fp32,
        )

    def _mix_from_attended_without_keep(
        self,
        attended_streams: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        dynamic_ab_weight = fused_weight[:, : self.num_streams * 2]
        alpha, beta = _dual_stream_alpha_beta_impl(
            attended_streams,
            dynamic_ab_weight,
            residual_state.static_alpha,
            residual_state.static_beta,
            residual_state.pre_branch_scale,
            residual_state.h_post_scale,
            residual_state.norm.eps,
            residual_state.norm.force_fp32,
        )
        stream0 = attended_streams[:, :, 0, :]
        stream1 = attended_streams[:, :, 1, :]
        branch_input = alpha[..., :1] * stream0 + alpha[..., 1:] * stream1
        return branch_input.to(dtype=attended_streams.dtype), stream0, stream1, beta, None

    def _keep_from_partial(
        self,
        partial_block: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
    ) -> torch.Tensor:
        source_streams = self._to_stream_tensor(partial_block)
        residual_weight = fused_weight[:, -1:]
        return _dual_stream_keep_impl(
            source_streams,
            residual_weight,
            residual_state.static_residual_logit,
            residual_state.residual_scale,
            residual_state.norm.eps,
            residual_state.norm.force_fp32,
            _DUAL_KEEP_SOURCE == "static",
        )

    def _alpha_beta_from_partial(
        self,
        partial_block: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_streams = self._to_stream_tensor(partial_block)
        dynamic_ab_weight = fused_weight[:, : self.num_streams * 2]
        return _dual_stream_alpha_beta_impl(
            source_streams,
            dynamic_ab_weight,
            residual_state.static_alpha,
            residual_state.static_beta,
            residual_state.pre_branch_scale,
            residual_state.h_post_scale,
            residual_state.norm.eps,
            residual_state.norm.force_fp32,
        )

    def _try_triton_fused_residual_read(
        self,
        partial_block: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
        history_prefix: Optional[tuple[torch.Tensor, torch.Tensor]],
        include_current: bool,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
        global _DUAL_TRITON_FUSED_RESIDUAL_READ_DISABLED, _DUAL_TRITON_FUSED_RESIDUAL_READ_GATE_DEBUG_PRINTED

        def gate_debug(reason: str) -> None:
            global _DUAL_TRITON_FUSED_RESIDUAL_READ_GATE_DEBUG_PRINTED
            if not _DUAL_TRITON_FUSED_RESIDUAL_READ_DEBUG or _DUAL_TRITON_FUSED_RESIDUAL_READ_GATE_DEBUG_PRINTED:
                return
            _DUAL_TRITON_FUSED_RESIDUAL_READ_GATE_DEBUG_PRINTED = True
            print(
                "[dual_triton_gate] "
                f"skip={reason} "
                f"enabled={_DUAL_TRITON_FUSED_RESIDUAL_READ} disabled={_DUAL_TRITON_FUSED_RESIDUAL_READ_DISABLED} "
                f"triton_func={triton_fused_dual_prefix_residual_read is not None} "
                f"triton_available={_dual_residual_triton_is_available()} "
                f"history_prefix={history_prefix is not None} include_current={include_current} "
                f"defer_stream_mix={_DUAL_DEFER_STREAM_MIX} current_partial={_DUAL_CURRENT_PARTIAL_RESIDUAL} "
                f"defer_partial_keep={_DUAL_DEFER_PARTIAL_KEEP_MIX} "
                f"ab={_DUAL_AB_SOURCE} keep={_DUAL_KEEP_SOURCE} current_key={_DUAL_CURRENT_KEY_IMPL} "
                f"kv={_DUAL_KV_MODE} attres_limit={_DUAL_ATTRES_LIMIT} "
                f"exclude_raw={_DUAL_EXCLUDE_RAW_HISTORY} rawless={_DUAL_RAWLESS_RESIDUAL} "
                f"is_cuda={partial_block.is_cuda}",
                flush=True,
            )

        if not _DUAL_TRITON_FUSED_RESIDUAL_READ:
            gate_debug("disabled_by_env")
            return None
        if _DUAL_TRITON_FUSED_RESIDUAL_READ_DISABLED:
            gate_debug("disabled_after_error")
            return None
        if triton_fused_dual_prefix_residual_read is None or not _dual_residual_triton_is_available():
            gate_debug("triton_unavailable")
            return None
        if history_prefix is None or not include_current:
            gate_debug("no_prefix_or_no_current")
            return None
        if not (
            _DUAL_DEFER_STREAM_MIX
            and _DUAL_CURRENT_PARTIAL_RESIDUAL
            and _DUAL_DEFER_PARTIAL_KEEP_MIX
            and _DUAL_AB_SOURCE == "attended"
            and _DUAL_KEEP_SOURCE == "partial"
            and _DUAL_CURRENT_KEY_IMPL == "stream_norm"
            and _DUAL_KV_MODE in {"crossk_selfv", "selfk_selfv"}
            and not _DUAL_ATTRES_LIMIT
            and not _DUAL_EXCLUDE_RAW_HISTORY
            and not _DUAL_RAWLESS_RESIDUAL
        ):
            gate_debug("condition_mismatch")
            return None
        if not partial_block.is_cuda:
            gate_debug("partial_not_cuda")
            return None
        gate_debug("launch")
        t0 = _dual_profile_now()
        current_streams = self._to_stream_tensor(partial_block)
        if _DUAL_TRITON_CURRENT_KEY_NORM:
            if triton_fixed_rms_norm is None:
                raise RuntimeError("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM requires Triton")
            current_key_streams = triton_fixed_rms_norm(
                current_streams,
                residual_state.history_key_norm.eps,
            )
        else:
            current_key_streams = residual_state.history_key_norm(current_streams)
        _dual_profile_record("current_prepare", t0)
        t0 = _dual_profile_now()
        try:
            branch_input, residual0, residual1, beta, keep = triton_fused_dual_prefix_residual_read(
                history_prefix,
                current_streams,
                current_key_streams,
                residual_state.q,
                fused_weight,
                residual_state.static_alpha,
                residual_state.static_beta,
                residual_state.static_residual_logit,
                residual_state.pre_branch_scale,
                residual_state.h_post_scale,
                residual_state.residual_scale,
                residual_state.norm.eps,
                cross_key=_DUAL_KV_MODE == "crossk_selfv",
            )
        except Exception as exc:
            if _DUAL_TRITON_FUSED_RESIDUAL_READ_STRICT:
                raise
            _DUAL_TRITON_FUSED_RESIDUAL_READ_DISABLED = True
            if _DUAL_TRITON_FUSED_RESIDUAL_READ_DEBUG:
                import traceback

                print(
                    "[dual_triton] debug "
                    f"partial={tuple(partial_block.shape)} "
                    f"prefix_logit={tuple(history_prefix[0].shape)} "
                    f"prefix_attended={tuple(history_prefix[1].shape)} "
                    f"query={tuple(residual_state.q.shape)} fused_weight={tuple(fused_weight.shape)} "
                    f"dtypes partial={partial_block.dtype} prefix={history_prefix[1].dtype} "
                    f"query={residual_state.q.dtype} fused={fused_weight.dtype}",
                    flush=True,
                )
                traceback.print_exc()
            print(f"[dual_triton] fused residual read disabled after runtime error: {type(exc).__name__}: {exc}", flush=True)
            return None
        _dual_profile_record("triton_fused_residual_read", t0)
        return branch_input, residual0, residual1, beta, keep, True

    def _dual_block_attn_res(
        self,
        cache: dict,
        partial_block: torch.Tensor,
        residual_state: _DualResidualSublayerState,
        include_current: bool = True,
        history_logits: Optional[torch.Tensor] = None,
        history_prefix: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        defer_partial_keep: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        t0 = _dual_profile_now()
        fused_weight = residual_state.fused_weight()
        history_keys = self._history_keys(cache, residual_state.history_key_norm)
        attention_cache = {"keys": history_keys, "values": cache["values"]}
        attention_include_current = include_current
        rawless_cache = None
        rawless_include_current = include_current
        if _DUAL_EXCLUDE_RAW_HISTORY or _DUAL_RAWLESS_RESIDUAL:
            keys = history_keys
            values = cache["values"]
            if keys is not None and values is not None:
                rawless_cache = {
                    "keys": keys[:, 1:],
                    "values": values[:, 1:],
                }
                if int(rawless_cache["keys"].shape[1]) == 0 and not rawless_include_current:
                    rawless_include_current = True
        if _DUAL_EXCLUDE_RAW_HISTORY and rawless_cache is not None:
            attention_cache = rawless_cache
            attention_include_current = rawless_include_current
            history_logits = None
            history_prefix = None
        fused = self._try_triton_fused_residual_read(
            partial_block,
            fused_weight,
            residual_state,
            history_prefix,
            attention_include_current,
        )
        if fused is not None:
            return fused
        attended = self._dual_history_attention(
            attention_cache,
            partial_block,
            residual_state.q,
            residual_state.history_key_norm,
            include_current=attention_include_current,
            history_logits=history_logits,
            history_prefix=history_prefix,
        )
        _dual_profile_record("history_attention", t0)
        t0 = _dual_profile_now()
        skip_attended_keep = (
            defer_partial_keep
            and _DUAL_DEFER_STREAM_MIX
            and _DUAL_AB_SOURCE == "attended"
            and _DUAL_KEEP_SOURCE == "partial"
        )
        if skip_attended_keep:
            mixed = self._mix_from_attended_without_keep(
                attended,
                fused_weight,
                residual_state,
            )
        else:
            mixed = self._mix_from_attended(
                attended,
                fused_weight,
                residual_state,
            )
        if _DUAL_DEFER_STREAM_MIX:
            if _DUAL_AB_SOURCE != "attended":
                alpha, beta = self._alpha_beta_from_partial(
                    partial_block,
                    fused_weight,
                    residual_state,
                )
                attended_stream0 = attended[:, :, 0, :]
                attended_stream1 = attended[:, :, 1, :]
                branch_input = alpha[..., :1] * attended_stream0 + alpha[..., 1:] * attended_stream1
                mixed = (branch_input, mixed[1], mixed[2], beta, mixed[4])
            if _DUAL_KEEP_SOURCE != "attended":
                keep = (
                    None
                    if skip_attended_keep
                    else self._keep_from_partial(partial_block, fused_weight, residual_state)
                )
                mixed = (mixed[0], mixed[1], mixed[2], mixed[3], keep)
        elif _DUAL_AB_SOURCE != "attended" or _DUAL_KEEP_SOURCE != "attended":
            raise ValueError(
                "MEGATRON_DUAL_AB_SOURCE=partial or MEGATRON_DUAL_KEEP_SOURCE=partial/static "
                "requires MEGATRON_DUAL_DEFER_STREAM_MIX=1"
            )
        _dual_profile_record("scalar_mix", t0)
        if _DUAL_RAWLESS_RESIDUAL and not _DUAL_EXCLUDE_RAW_HISTORY and rawless_cache is not None:
            t0 = _dual_profile_now()
            rawless_attended = self._dual_history_attention(
                rawless_cache,
                partial_block,
                residual_state.q,
                residual_state.history_key_norm,
                include_current=rawless_include_current,
                history_logits=None,
                history_prefix=None,
            )
            _dual_profile_record("rawless_residual_attention", t0)
            if _DUAL_DEFER_STREAM_MIX:
                return (
                    mixed[0],
                    rawless_attended[:, :, 0, :],
                    rawless_attended[:, :, 1, :],
                    mixed[3],
                    mixed[4],
                )
            rawless_mixed = self._mix_from_attended(
                rawless_attended,
                fused_weight,
                residual_state,
            )
            return mixed[0], rawless_mixed[1], rawless_mixed[2], mixed[3]
        return mixed

    def _merge_delta(
        self,
        residual_stream0: torch.Tensor,
        residual_stream1: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(dtype=residual_stream0.dtype)
        if _DUAL_MERGE_IMPL in {"addcmul", "lerp_addcmul"}:
            merged0 = torch.addcmul(residual_stream0, delta, beta[..., :1])
            merged1 = torch.addcmul(residual_stream1, delta, beta[..., 1:])
        else:
            merged0 = residual_stream0 + delta * beta[..., :1]
            merged1 = residual_stream1 + delta * beta[..., 1:]
        merged = torch.stack((merged0, merged1), dim=2)
        return self._from_stream_tensor(merged)

    def _merge_delta_deferred(
        self,
        stream0: torch.Tensor,
        stream1: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(dtype=stream0.dtype)
        if _DUAL_TRITON_DEFERRED_MERGE:
            if triton_dual_deferred_merge is None or not _dual_residual_triton_is_available():
                raise RuntimeError("MEGATRON_DUAL_TRITON_DEFERRED_MERGE=1 requires Triton")
            merged = triton_dual_deferred_merge(stream0, stream1, delta, beta, keep)
            return self._from_stream_tensor(merged)
        keep = keep.unsqueeze(-1).to(dtype=stream0.dtype)
        if _DUAL_MERGE_IMPL == "lerp_addcmul":
            mixed0 = torch.lerp(stream1, stream0, keep)
            mixed1 = torch.lerp(stream0, stream1, keep)
        else:
            one_minus_keep = 1.0 - keep
            mixed0 = keep * stream0 + one_minus_keep * stream1
            mixed1 = one_minus_keep * stream0 + keep * stream1
        if _DUAL_MERGE_IMPL in {"addcmul", "lerp_addcmul"}:
            merged0 = torch.addcmul(mixed0, delta, beta[..., :1])
            merged1 = torch.addcmul(mixed1, delta, beta[..., 1:])
        else:
            merged0 = mixed0 + delta * beta[..., :1]
            merged1 = mixed1 + delta * beta[..., 1:]
        merged = torch.stack((merged0, merged1), dim=2)
        return self._from_stream_tensor(merged)

    def _mix_current_residual(self, partial_block: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = self._to_stream_tensor(partial_block)
        keep = keep.unsqueeze(-1).to(dtype=streams.dtype)
        stream0 = streams[:, :, 0, :]
        stream1 = streams[:, :, 1, :]
        if _DUAL_MERGE_IMPL == "lerp_addcmul":
            mixed0 = torch.lerp(stream1, stream0, keep)
            mixed1 = torch.lerp(stream0, stream1, keep)
        else:
            one_minus_keep = 1.0 - keep
            mixed0 = keep * stream0 + one_minus_keep * stream1
            mixed1 = one_minus_keep * stream0 + keep * stream1
        return mixed0, mixed1

    def _mix_current_residual_from_partial(
        self,
        partial_block: torch.Tensor,
        fused_weight: torch.Tensor,
        residual_state: _DualResidualSublayerState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = self._to_stream_tensor(partial_block)
        residual_weight = fused_weight[:, -1:]
        return _dual_stream_keep_mix_current_impl(
            streams,
            residual_weight,
            residual_state.static_residual_logit,
            residual_state.residual_scale,
            residual_state.norm.eps,
            residual_state.norm.force_fp32,
            _DUAL_KEEP_SOURCE == "static",
        )

    def _merge_delta_only(self, delta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        delta = delta.to(dtype=beta.dtype)
        merged0 = delta * beta[..., :1]
        merged1 = delta * beta[..., 1:]
        merged = torch.stack((merged0, merged1), dim=2)
        return self._from_stream_tensor(merged)

    def _merge_current_delta(self, partial_block: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        streams = self._to_stream_tensor(partial_block)
        delta = delta.to(dtype=streams.dtype)
        beta = beta.to(dtype=streams.dtype)
        merged0 = streams[:, :, 0, :] + delta * beta[..., :1]
        merged1 = streams[:, :, 1, :] + delta * beta[..., 1:]
        merged = torch.stack((merged0, merged1), dim=2)
        return self._from_stream_tensor(merged)

    @staticmethod
    def _zero_dependency(reference: torch.Tensor, *tensors: Optional[torch.Tensor]) -> torch.Tensor:
        zero = reference.new_zeros(())
        for tensor in tensors:
            if tensor is not None:
                zero = zero + tensor.to(dtype=reference.dtype).sum() * 0.0
        return zero

    def forward(self, x: torch.Tensor, *, attention_mask=None, rotary_pos_emb=None, state: Optional[dict] = None) -> torch.Tensor:
        if state is None:
            raise ValueError("DualAttentionResidualBlock requires a mutable state dict")
        t0 = _dual_profile_now()
        cache = self._get_cache(x, state)
        _dual_profile_record("cache", t0)
        partial_block = x
        aux_loss = None
        block_started = False
        history_includes_current_input = bool(state.pop(self._history_includes_input_key, False))
        skip_initial_embed_append = (
            _DUAL_EMBED_AS_PARTIAL
            and not bool(cache["blocks"])
            and not bool(state.get(self._embed_as_partial_consumed_key, False))
        )
        history_logits_cache: dict[tuple[int, str], torch.Tensor] = {}
        history_prefix_cache: dict[
            tuple[int, str], Optional[tuple[torch.Tensor, torch.Tensor]]
        ] = {}

        def get_history_prefix(
            layer_index: int,
            name: str,
            residual_state: _DualResidualSublayerState,
            include_current: bool,
        ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
            if not _DUAL_CACHE_HISTORY_PREFIX:
                return None
            keys = self._history_keys(cache, residual_state.history_key_norm)
            values = cache["values"]
            if keys is None or values is None:
                return None
            if not include_current and int(keys.shape[1]) == 1:
                return None
            cache_key = (layer_index, name)
            if cache_key in history_prefix_cache:
                return history_prefix_cache[cache_key]
            attn_history_norm = self.attn_states[layer_index].history_key_norm
            mlp_history_norm = self.mlp_states[layer_index].history_key_norm
            pair_has_equivalent_norms = (
                self._history_norm_cache_key(attn_history_norm)
                == self._history_norm_cache_key(mlp_history_norm)
            )
            if _DUAL_CACHE_HISTORY_PREFIX_PAIR and pair_has_equivalent_norms:
                t0 = _dual_profile_now()
                attn_prefix, mlp_prefix = _dual_history_prefix_pair(
                    keys,
                    values.to(dtype=keys.dtype),
                    self.attn_states[layer_index].q,
                    self.mlp_states[layer_index].q,
                    history_blocks=(
                        cache["blocks"]
                        if _DUAL_RECOMPUTE_HISTORY_VALUES or _DUAL_RECOMPUTE_HISTORY_KEYS
                        else None
                    ),
                    history_norm_eps=residual_state.history_key_norm.eps,
                    history_norm_fp32=residual_state.history_key_norm.force_fp32,
                )
                _dual_profile_record("history_prefix_pair_precompute", t0)
                history_prefix_cache[(layer_index, "attn")] = attn_prefix
                history_prefix_cache[(layer_index, "mlp")] = mlp_prefix
                return history_prefix_cache[cache_key]
            t0 = _dual_profile_now()
            prefix = _dual_history_prefix(
                keys,
                values.to(dtype=keys.dtype),
                residual_state.q,
            )
            _dual_profile_record("history_prefix_precompute", t0)
            history_prefix_cache[cache_key] = prefix
            return prefix

        def get_history_logits(
            layer_index: int,
            name: str,
            residual_state: _DualResidualSublayerState,
            include_current: bool,
        ) -> Optional[torch.Tensor]:
            if _DUAL_CACHE_HISTORY_PREFIX or not _DUAL_CACHE_HISTORY_LOGITS:
                return None
            keys = self._history_keys(cache, residual_state.history_key_norm)
            if keys is None:
                return None
            if not include_current and int(keys.shape[1]) == 1:
                return None
            cache_key = (layer_index, name)
            cached = history_logits_cache.get(cache_key)
            if cached is not None and int(cached.shape[1]) == int(keys.shape[1]):
                return cached
            t0 = _dual_profile_now()
            logits = _dual_history_logits(keys, residual_state.q.to(dtype=keys.dtype))
            _dual_profile_record("history_logits_precompute", t0)
            history_logits_cache[cache_key] = logits
            return logits

        for layer_index, layer in enumerate(self.layers):
            attn_state = self.attn_states[layer_index]
            mlp_state = self.mlp_states[layer_index]
            attn_fused_weight = attn_state.fused_weight()
            mlp_fused_weight = mlp_state.fused_weight()
            current_is_already_in_history = history_includes_current_input or (
                bool(cache["blocks"]) and cache["blocks"][-1] is partial_block
            )
            if not block_started and not current_is_already_in_history:
                if not skip_initial_embed_append:
                    t0 = _dual_profile_now()
                    self._append_block(cache, partial_block)
                    _dual_profile_record("append_history", t0)
                    history_logits_cache.clear()
                    history_prefix_cache.clear()
                    current_is_already_in_history = True
            include_current = not (not block_started and current_is_already_in_history)
            attn_history_prefix = get_history_prefix(
                layer_index,
                "attn",
                attn_state,
                include_current,
            )
            first_attn_in_block = not block_started
            t0 = _dual_profile_now()
            attn_mix = self._dual_block_attn_res(
                cache,
                partial_block,
                attn_state,
                include_current=include_current,
                history_logits=get_history_logits(
                    layer_index,
                    "attn",
                    attn_state,
                    include_current,
                ),
                history_prefix=attn_history_prefix,
                defer_partial_keep=_DUAL_DEFER_PARTIAL_KEEP_MIX,
            )
            if _DUAL_DEFER_STREAM_MIX:
                attn_current_residual_ready = False
                if len(attn_mix) == 6:
                    attn_hidden, attn_residual0, attn_residual1, attn_beta, attn_keep, attn_current_residual_ready = attn_mix
                else:
                    attn_hidden, attn_residual0, attn_residual1, attn_beta, attn_keep = attn_mix
            else:
                attn_hidden, attn_residual0, attn_residual1, attn_beta = attn_mix
                attn_keep = None
                attn_current_residual_ready = False
            _dual_profile_record("attn_residual_read", t0)
            if not block_started:
                block_started = True
                history_includes_current_input = False
            t0 = _dual_profile_now()
            raw_attn_output = None
            if _DUAL_TE_ATTENTION_EPILOGUE:
                if not _DUAL_CURRENT_PARTIAL_RESIDUAL and not self.full_history:
                    raise ValueError("Dual TE attention epilogue requires current-partial residual mode")
                if _DUAL_ATTRES_LIMIT:
                    raise ValueError("Dual TE attention epilogue does not support AttnRes-limit mode")
                if triton_dual_te_attention_epilogue is None or not _dual_residual_triton_is_available():
                    raise RuntimeError("MEGATRON_DUAL_TE_ATTENTION_EPILOGUE=1 requires Triton")
                raw_attn_output = layer.attention_raw_output(
                    attn_hidden,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )
                if self.full_history:
                    attn_delta = triton_te_attention_delta(attn_hidden, raw_attn_output)
                    raw_attn_output = None
                else:
                    attn_delta = None
            else:
                attn_delta = _residual_attention_delta(
                    layer,
                    attn_hidden,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )
            _dual_profile_record("attention_delta", t0)
            t0 = _dual_profile_now()
            if self.full_history:
                if _DUAL_DEFER_STREAM_MIX:
                    if attn_keep is None:
                        attn_keep = self._keep_from_partial(
                            partial_block,
                            attn_fused_weight,
                            attn_state,
                        )
                    partial_block = self._merge_delta_deferred(
                        attn_residual0,
                        attn_residual1,
                        attn_delta,
                        attn_beta,
                        attn_keep,
                    )
                else:
                    partial_block = self._merge_delta(attn_residual0, attn_residual1, attn_delta, attn_beta)
            elif _DUAL_ATTRES_LIMIT:
                if first_attn_in_block:
                    partial_block = self._merge_delta_only(attn_delta, attn_beta)
                else:
                    partial_block = self._merge_current_delta(partial_block, attn_delta, attn_beta)
                partial_block = partial_block + self._zero_dependency(
                    partial_block,
                    attn_residual0,
                    attn_residual1,
                    attn_keep,
                )
            elif _DUAL_CURRENT_PARTIAL_RESIDUAL:
                if not _DUAL_DEFER_STREAM_MIX:
                    raise ValueError("MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL requires MEGATRON_DUAL_DEFER_STREAM_MIX=1")
                if first_attn_in_block:
                    if raw_attn_output is None:
                        partial_block = self._merge_delta_only(attn_delta, attn_beta)
                    else:
                        merged = triton_dual_te_attention_epilogue(
                            attn_hidden,
                            raw_attn_output,
                            attn_residual0,
                            attn_residual1,
                            attn_beta,
                            has_residual=False,
                        )
                        partial_block = self._from_stream_tensor(merged)
                    # Equation (15) intentionally omits rho for the first
                    # branch.  These zero dependencies preserve that exact
                    # forward while satisfying Megatron DDP's requirement
                    # that every bucket parameter fire a grad-ready hook.
                    partial_block = partial_block + self._zero_dependency(
                        partial_block,
                        attn_state.dynamic_residual_mix_fn,
                        attn_state.static_residual_logit,
                        attn_state.residual_scale,
                    )
                else:
                    if attn_current_residual_ready:
                        current_residual0, current_residual1 = attn_residual0, attn_residual1
                    elif attn_keep is None:
                        if not _DUAL_DEFER_PARTIAL_KEEP_MIX:
                            raise ValueError(
                                "MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL requires a keep tensor "
                                "unless MEGATRON_DUAL_FUSE_PARTIAL_KEEP_MIX=1"
                            )
                        current_residual0, current_residual1, attn_keep = self._mix_current_residual_from_partial(
                            partial_block,
                            attn_fused_weight,
                            attn_state,
                        )
                    else:
                        current_residual0, current_residual1 = self._mix_current_residual(partial_block, attn_keep)
                    if raw_attn_output is None:
                        partial_block = self._merge_delta(
                            current_residual0,
                            current_residual1,
                            attn_delta,
                            attn_beta,
                        )
                    else:
                        merged = triton_dual_te_attention_epilogue(
                            attn_hidden,
                            raw_attn_output,
                            current_residual0,
                            current_residual1,
                            attn_beta,
                            has_residual=True,
                        )
                        partial_block = self._from_stream_tensor(merged)
            elif _DUAL_ATTRES_LIKE_FIRST_ATTN and first_attn_in_block:
                partial_block = self._merge_delta_only(attn_delta, attn_beta)
            elif _DUAL_DEFER_STREAM_MIX:
                partial_block = self._merge_delta_deferred(
                    attn_residual0,
                    attn_residual1,
                    attn_delta,
                    attn_beta,
                    attn_keep,
                )
            else:
                partial_block = self._merge_delta(attn_residual0, attn_residual1, attn_delta, attn_beta)
            _dual_profile_record("attn_merge", t0)

            t0 = _dual_profile_now()
            mlp_history_prefix = get_history_prefix(layer_index, "mlp", mlp_state, True)
            mlp_mix = self._dual_block_attn_res(
                cache,
                partial_block,
                mlp_state,
                history_logits=get_history_logits(layer_index, "mlp", mlp_state, True),
                history_prefix=mlp_history_prefix,
                defer_partial_keep=_DUAL_DEFER_PARTIAL_KEEP_MIX,
            )
            if _DUAL_DEFER_STREAM_MIX:
                mlp_current_residual_ready = False
                if len(mlp_mix) == 6:
                    mlp_hidden, mlp_residual0, mlp_residual1, mlp_beta, mlp_keep, mlp_current_residual_ready = mlp_mix
                else:
                    mlp_hidden, mlp_residual0, mlp_residual1, mlp_beta, mlp_keep = mlp_mix
            else:
                mlp_hidden, mlp_residual0, mlp_residual1, mlp_beta = mlp_mix
                mlp_keep = None
                mlp_current_residual_ready = False
            _dual_profile_record("mlp_residual_read", t0)
            if self.full_history:
                self._append_block(cache, partial_block)
                history_logits_cache.clear()
                history_prefix_cache.clear()
            t0 = _dual_profile_now()
            mlp_delta = layer.mlp_delta(mlp_hidden)
            _dual_profile_record("mlp_delta", t0)
            t0 = _dual_profile_now()
            if self.full_history:
                if _DUAL_DEFER_STREAM_MIX:
                    if mlp_keep is None:
                        mlp_keep = self._keep_from_partial(
                            partial_block,
                            mlp_fused_weight,
                            mlp_state,
                        )
                    partial_block = self._merge_delta_deferred(
                        mlp_residual0,
                        mlp_residual1,
                        mlp_delta,
                        mlp_beta,
                        mlp_keep,
                    )
                else:
                    partial_block = self._merge_delta(mlp_residual0, mlp_residual1, mlp_delta, mlp_beta)
            elif _DUAL_ATTRES_LIMIT:
                partial_block = self._merge_current_delta(partial_block, mlp_delta, mlp_beta)
                partial_block = partial_block + self._zero_dependency(
                    partial_block,
                    mlp_residual0,
                    mlp_residual1,
                    mlp_keep,
                )
            elif _DUAL_CURRENT_PARTIAL_RESIDUAL:
                if not _DUAL_DEFER_STREAM_MIX:
                    raise ValueError("MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL requires MEGATRON_DUAL_DEFER_STREAM_MIX=1")
                if mlp_current_residual_ready:
                    current_residual0, current_residual1 = mlp_residual0, mlp_residual1
                elif mlp_keep is None:
                    if not _DUAL_DEFER_PARTIAL_KEEP_MIX:
                        raise ValueError(
                            "MEGATRON_DUAL_CURRENT_PARTIAL_RESIDUAL requires a keep tensor "
                            "unless MEGATRON_DUAL_FUSE_PARTIAL_KEEP_MIX=1"
                        )
                    current_residual0, current_residual1, mlp_keep = self._mix_current_residual_from_partial(
                        partial_block,
                        mlp_fused_weight,
                        mlp_state,
                    )
                else:
                    current_residual0, current_residual1 = self._mix_current_residual(partial_block, mlp_keep)
                partial_block = self._merge_delta(current_residual0, current_residual1, mlp_delta, mlp_beta)
            elif _DUAL_DEFER_STREAM_MIX:
                partial_block = self._merge_delta_deferred(
                    mlp_residual0,
                    mlp_residual1,
                    mlp_delta,
                    mlp_beta,
                    mlp_keep,
                )
            else:
                partial_block = self._merge_delta(mlp_residual0, mlp_residual1, mlp_delta, mlp_beta)
            _dual_profile_record("mlp_merge", t0)
            if self.full_history:
                self._append_block(cache, partial_block)
                history_logits_cache.clear()
                history_prefix_cache.clear()
            if layer.last_aux_loss is not None:
                aux_loss = layer.last_aux_loss if aux_loss is None else aux_loss + layer.last_aux_loss
        state[self._history_key] = cache["blocks"]
        if skip_initial_embed_append:
            state[self._embed_as_partial_consumed_key] = True
        self.last_aux_loss = aux_loss
        return partial_block


class DualAttentionResidualOutputAggregation(nn.Module):
    _history_key = "history"
    _cache_key = "history_cache"

    def __init__(self, config) -> None:
        super().__init__()
        self.num_streams = 2
        dual_norm_fp32 = _env_bool("MEGATRON_DUAL_RMSNORM_FP32", True)
        self.history_key_norm = FixedRMSNorm(config.model.layernorm_epsilon, force_fp32=dual_norm_fp32)
        self.q = nn.Parameter(torch.empty(self.num_streams, config.model.hidden_size, dtype=_train_param_dtype(config)))
        init_dual_stream_query_(self.q, preferred_stream=0, balanced=_DUAL_Q_BALANCED)
        _mark_sequence_parallel_parameters(self.q)

    def _to_stream_tensor(self, residuals: torch.Tensor) -> torch.Tensor:
        seq, batch2, hidden = residuals.shape
        return residuals.view(seq, batch2 // self.num_streams, self.num_streams, hidden)

    def _from_stream_tensor(self, residuals: torch.Tensor) -> torch.Tensor:
        seq, batch, streams, hidden = residuals.shape
        return residuals.contiguous().view(seq, batch * streams, hidden)

    def _pack(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = self._to_stream_tensor(value)
        keys = self.history_key_norm(streams).transpose(0, 2)
        if _DUAL_KV_MODE == "crossk_selfv":
            keys = keys.flip(0)
        values = streams.transpose(0, 2)
        if _DUAL_KV_MODE == "selfk_crossv":
            values = values.flip(0)
        keys = keys.contiguous()
        values = values.contiguous()
        return keys, values

    def _history_blocks(self, state: dict) -> list[torch.Tensor]:
        blocks = state.get(self._history_key)
        if isinstance(blocks, torch.Tensor):
            return [blocks]
        if isinstance(blocks, tuple):
            return list(blocks)
        if isinstance(blocks, list):
            return blocks
        return []

    def _pack_final_history(
        self, hidden_states: torch.Tensor, state: dict
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        blocks = self._history_blocks(state)
        cache = state.get(self._cache_key)
        cached_keys = None
        if cache is not None:
            cache_key = _dual_history_norm_cache_key(self.history_key_norm)
            cache_entry = cache.get("keys_by_norm", {}).get(cache_key)
            if cache_entry is not None:
                cached_keys = cache_entry.get("keys")
        if (
            cache is not None
            and cached_keys is not None
            and cache.get("values") is not None
            and int(cached_keys.shape[1]) == len(blocks)
        ):
            # Normally the block cache stores [raw_input, block1_input, ...],
            # and final aggregation drops raw_input.  In embed-as-partial mode,
            # raw_input was never cached, so keep every completed block.
            first_history_index = 0 if _DUAL_EMBED_AS_PARTIAL else 1
            return (
                cached_keys[:, first_history_index:],
                cache["values"][:, first_history_index:],
                *self._pack(hidden_states),
            )

        first_history_index = 0 if _DUAL_EMBED_AS_PARTIAL else 1
        history_values = [*blocks[first_history_index:], hidden_states]
        keys, values = zip(*(self._pack(block) for block in history_values))
        return torch.stack(keys, dim=1), torch.stack(values, dim=1), None, None

    def forward(self, hidden_states: torch.Tensor, state: dict) -> torch.Tensor:
        t0 = _dual_profile_now()
        packed_keys, packed_values, current_key, current_value = self._pack_final_history(hidden_states, state)
        _dual_profile_record("output_pack", t0)
        t0 = _dual_profile_now()
        attended = _dual_depth_attention(
            packed_keys,
            packed_values,
            self.q,
            current_key=current_key,
            current_value=current_value,
            output_dtype=hidden_states.dtype,
        )
        _dual_profile_record("output_logits_softmax", t0)
        return self._from_stream_tensor(attended)
