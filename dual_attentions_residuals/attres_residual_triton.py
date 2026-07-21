from __future__ import annotations

import os
from typing import Optional

import torch
from torch.autograd import Function

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - Triton is optional on CPU/dev hosts.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _pow2_block(value: int, *, default: int, minimum: int, maximum: int) -> int:
    value = int(value)
    if value <= 0:
        value = int(default)
    value = max(int(minimum), min(int(maximum), value))
    return int(1 << (value - 1).bit_length())


if _TRITON_AVAILABLE:

    @triton.jit
    def _round_bf16_fp32(value):
        rounded = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;",
            "=h,f",
            [value],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )
        return rounded.to(tl.float32)

    @triton.jit
    def _attres_history_append_prefix_copy_kernel(
        prev_keys,
        prev_values,
        keys_out,
        values_out,
        prefix_elements: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < prefix_elements
        key = tl.load(prev_keys + offsets, mask=mask, other=0.0)
        value = tl.load(prev_values + offsets, mask=mask, other=0.0)
        tl.store(keys_out + offsets, key, mask=mask)
        tl.store(values_out + offsets, value, mask=mask)

    @triton.jit
    def _attres_history_append_new_forward_kernel(
        block,
        keys_out,
        values_out,
        inv_rms_out,
        output_history_size: tl.constexpr,
        rows: tl.constexpr,
        hidden_size: tl.constexpr,
        eps,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        input_base = row * hidden_size
        output_base = ((output_history_size - 1) * rows + row) * hidden_size
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            value = tl.load(block + input_base + hidden, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            value = tl.load(block + input_base + hidden, mask=mask, other=0.0).to(tl.float32)
            tl.store(keys_out + output_base + hidden, value * inv_rms, mask=mask)
            tl.store(values_out + output_base + hidden, value, mask=mask)
        tl.store(inv_rms_out + row, inv_rms)

    @triton.jit
    def _attres_history_append_new_backward_kernel(
        grad_keys,
        grad_values,
        block,
        inv_rms,
        grad_block,
        output_history_size: tl.constexpr,
        rows: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        input_base = row * hidden_size
        output_base = ((output_history_size - 1) * rows + row) * hidden_size
        inv = tl.load(inv_rms + row)
        dot = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            value = tl.load(block + input_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_key = tl.load(grad_keys + output_base + hidden, mask=mask, other=0.0).to(tl.float32)
            dot += grad_key * value
        correction = tl.sum(dot, axis=0) * inv * inv * inv / hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            value = tl.load(block + input_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_key = tl.load(grad_keys + output_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_value = tl.load(grad_values + output_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_from_key = _round_bf16_fp32(grad_key * inv - value * correction)
            grad = _round_bf16_fp32(grad_value + grad_from_key)
            tl.store(grad_block + input_base + hidden, grad, mask=mask)

    @triton.jit
    def _attres_current_finalize_forward_kernel(
        history_logit,
        history_attended,
        current,
        current_logit,
        attended_out,
        history_weight_out,
        current_weight_out,
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        history_stride_seq: tl.constexpr,
        history_stride_batch: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        row_base = row * hidden_size
        current_logit_value = tl.load(current_logit + row).to(tl.float32)
        seq_idx = row // batch_size
        batch_idx = row - seq_idx * batch_size
        history_logit_value = tl.load(
            history_logit + seq_idx * history_stride_seq + batch_idx * history_stride_batch
        ).to(tl.float32)
        history_delta = _round_bf16_fp32(history_logit_value - current_logit_value)
        current_delta = _round_bf16_fp32(current_logit_value - history_logit_value)
        history_weight = _round_bf16_fp32(tl.sigmoid(history_delta))
        current_weight = _round_bf16_fp32(tl.sigmoid(current_delta))
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            history_value = tl.load(
                history_attended + row_base + hidden, mask=mask, other=0.0
            ).to(tl.float32)
            current_value = tl.load(current + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            history_part = _round_bf16_fp32(history_value * history_weight)
            current_part = _round_bf16_fp32(current_value * current_weight)
            attended = _round_bf16_fp32(history_part + current_part)
            tl.store(attended_out + row_base + hidden, attended, mask=mask)
        tl.store(history_weight_out + row, history_weight)
        tl.store(current_weight_out + row, current_weight)

    @triton.jit
    def _attres_current_finalize_backward_row_kernel(
        grad_attended,
        history_attended,
        current,
        history_weight,
        current_weight,
        grad_history_logit_out,
        grad_history_attended_out,
        grad_current_out,
        grad_current_logit_out,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        row_base = row * hidden_size
        hw = tl.load(history_weight + row).to(tl.float32)
        cw = tl.load(current_weight + row).to(tl.float32)
        grad_hw = tl.zeros((block_h,), dtype=tl.float32)
        grad_cw = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            grad = tl.load(grad_attended + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            history_value = tl.load(
                history_attended + row_base + hidden, mask=mask, other=0.0
            ).to(tl.float32)
            current_value = tl.load(current + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_hw += _round_bf16_fp32(grad * history_value)
            grad_cw += _round_bf16_fp32(grad * current_value)
        grad_hw_value = _round_bf16_fp32(tl.sum(grad_hw, axis=0))
        grad_cw_value = _round_bf16_fp32(tl.sum(grad_cw, axis=0))
        one_minus_hw = _round_bf16_fp32(1.0 - hw)
        one_minus_cw = _round_bf16_fp32(1.0 - cw)
        grad_history_from_hw = _round_bf16_fp32(
            _round_bf16_fp32(grad_hw_value * one_minus_hw) * hw
        )
        grad_history_from_cw = _round_bf16_fp32(
            _round_bf16_fp32(grad_cw_value * one_minus_cw) * cw
        )
        grad_history_logit = _round_bf16_fp32(grad_history_from_hw - grad_history_from_cw)
        grad_current_logit = _round_bf16_fp32(-grad_history_logit)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + offsets
            mask = hidden < hidden_size
            grad = tl.load(grad_attended + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            grad_history_value = _round_bf16_fp32(grad * hw)
            grad_current_value = _round_bf16_fp32(grad * cw)
            tl.store(
                grad_history_attended_out + row_base + hidden, grad_history_value, mask=mask
            )
            tl.store(grad_current_out + row_base + hidden, grad_current_value, mask=mask)
        tl.store(grad_history_logit_out + row, grad_history_logit)
        tl.store(grad_current_logit_out + row, grad_current_logit)

    @triton.jit
    def _attres_weighted_sum_forward_kernel(
        weights,
        values,
        attended_out,
        seq_batch: tl.constexpr,
        history_size: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, block_h)
        output_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            hidden_mask = hidden < hidden_size
            attended = tl.zeros((block_h,), dtype=tl.float32)
            for history_idx in range(0, history_size):
                weight = tl.load(weights + history_idx * seq_batch + row).to(tl.float32)
                value_offsets = (history_idx * seq_batch + row) * hidden_size + hidden
                value = tl.load(values + value_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
                attended += weight * value
            tl.store(attended_out + output_base + hidden, attended, mask=hidden_mask)


    @triton.jit
    def _attres_weighted_sum_values_backward_kernel(
        weights,
        grad_attended,
        grad_values_out,
        seq_batch: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        entry = tl.program_id(0)
        history_idx = entry // seq_batch
        row = entry - history_idx * seq_batch
        weight = tl.load(weights + history_idx * seq_batch + row).to(tl.float32)
        h = tl.arange(0, block_h)
        output_base = row * hidden_size
        value_base = entry * hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            hidden_mask = hidden < hidden_size
            grad = tl.load(grad_attended + output_base + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            tl.store(grad_values_out + value_base + hidden, weight * grad, mask=hidden_mask)


def triton_is_available() -> bool:
    return bool(_TRITON_AVAILABLE)


class _TritonAttresHistoryAppendPack(Function):
    @staticmethod
    def forward(
        ctx,
        prev_keys: torch.Tensor,
        prev_values: torch.Tensor,
        block: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_size = int(prev_keys.shape[0])
        seq_len, batch_size, hidden_size = [int(value) for value in block.shape]
        output_history_size = history_size + 1
        output_shape = (output_history_size, seq_len, batch_size, hidden_size)
        keys = torch.empty(output_shape, device=block.device, dtype=block.dtype)
        values = torch.empty_like(keys)
        rows = seq_len * batch_size
        inv_rms = torch.empty((rows,), device=block.device, dtype=torch.float32)

        if history_size:
            prefix_elements = int(prev_keys.numel())
            copy_block = _pow2_block(
                _env_int("MEGATRON_ATTRES_TRITON_HISTORY_APPEND_COPY_BLOCK", 256),
                default=256,
                minimum=32,
                maximum=4096,
            )
            _attres_history_append_prefix_copy_kernel[(triton.cdiv(prefix_elements, copy_block),)](
                prev_keys,
                prev_values,
                keys,
                values,
                prefix_elements=prefix_elements,
                block=copy_block,
                num_warps=4,
            )

        block_h = _pow2_block(
            _env_int("MEGATRON_ATTRES_TRITON_HISTORY_APPEND_BLOCK_H", 128),
            default=128,
            minimum=16,
            maximum=max(16, hidden_size),
        )
        num_warps = _env_int("MEGATRON_ATTRES_TRITON_HISTORY_APPEND_NUM_WARPS", 4)
        _attres_history_append_new_forward_kernel[(rows,)](
            block,
            keys,
            values,
            inv_rms,
            output_history_size=output_history_size,
            rows=rows,
            hidden_size=hidden_size,
            eps=float(eps),
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.history_size = history_size
        ctx.output_history_size = output_history_size
        ctx.rows = rows
        ctx.hidden_size = hidden_size
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.save_for_backward(block, inv_rms)
        return keys, values

    @staticmethod
    def backward(
        ctx,
        grad_keys: Optional[torch.Tensor],
        grad_values: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        block, inv_rms = ctx.saved_tensors
        if grad_keys is None:
            grad_keys = torch.zeros(
                (ctx.output_history_size, ctx.rows, ctx.hidden_size),
                device=block.device,
                dtype=block.dtype,
            )
        else:
            grad_keys = grad_keys.contiguous()
        if grad_values is None:
            grad_values = torch.zeros_like(grad_keys)
        else:
            grad_values = grad_values.contiguous()
        grad_block = torch.empty_like(block)
        _attres_history_append_new_backward_kernel[(ctx.rows,)](
            grad_keys,
            grad_values,
            block,
            inv_rms,
            grad_block,
            output_history_size=ctx.output_history_size,
            rows=ctx.rows,
            hidden_size=ctx.hidden_size,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_keys[: ctx.history_size], grad_values[: ctx.history_size], grad_block, None


def triton_attres_history_append_pack(
    prev_keys: torch.Tensor,
    prev_values: torch.Tensor,
    block: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (prev_keys.is_cuda and prev_values.is_cuda and block.is_cuda):
        raise ValueError("triton_attres_history_append_pack requires CUDA tensors")
    if prev_keys.ndim != 4 or prev_values.shape != prev_keys.shape:
        raise ValueError("expected matching previous key/value tensors [history, seq, batch, hidden]")
    if block.ndim != 3 or tuple(int(value) for value in prev_keys.shape[1:]) != tuple(
        int(value) for value in block.shape
    ):
        raise ValueError("previous key/value layout does not match block [seq, batch, hidden]")
    if not (prev_keys.dtype == prev_values.dtype == block.dtype == torch.bfloat16):
        raise ValueError("triton_attres_history_append_pack currently requires BF16 tensors")
    if not (prev_keys.is_contiguous() and prev_values.is_contiguous() and block.is_contiguous()):
        raise ValueError("triton_attres_history_append_pack requires contiguous tensors")
    return _TritonAttresHistoryAppendPack.apply(prev_keys, prev_values, block, float(eps))


class _TritonAttresCurrentFinalize(Function):
    @staticmethod
    def forward(
        ctx,
        history_logit: torch.Tensor,
        history_attended: torch.Tensor,
        current: torch.Tensor,
        current_logit: torch.Tensor,
    ) -> torch.Tensor:
        seq_len, batch_size, hidden_size = [int(value) for value in current.shape]
        rows = seq_len * batch_size
        attended = torch.empty_like(current)
        history_weight = torch.empty((rows,), device=current.device, dtype=current.dtype)
        current_weight = torch.empty_like(history_weight)
        block_h = _pow2_block(
            _env_int("MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_BLOCK_H", 512),
            default=512,
            minimum=32,
            maximum=min(512, max(32, hidden_size)),
        )
        num_warps = _env_int("MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_NUM_WARPS", 4)
        _attres_current_finalize_forward_kernel[(rows,)](
            history_logit,
            history_attended,
            current,
            current_logit,
            attended,
            history_weight,
            current_weight,
            batch_size=batch_size,
            hidden_size=hidden_size,
            history_stride_seq=int(history_logit.stride(0)),
            history_stride_batch=int(history_logit.stride(1)),
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.rows = rows
        ctx.hidden_size = hidden_size
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.save_for_backward(history_attended, current, history_weight, current_weight)
        return attended

    @staticmethod
    def backward(
        ctx,
        grad_attended: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_attended, current, history_weight, current_weight = ctx.saved_tensors
        if grad_attended is None:
            grad_attended = torch.zeros_like(current)
        else:
            grad_attended = grad_attended.contiguous()
        grad_history_logit = torch.empty(
            current.shape[:-1], device=current.device, dtype=current.dtype
        )
        grad_history_attended = torch.empty_like(history_attended)
        grad_current = torch.empty_like(current)
        grad_current_logit = torch.empty(
            current.shape[:-1], device=current.device, dtype=current.dtype
        )
        _attres_current_finalize_backward_row_kernel[(ctx.rows,)](
            grad_attended,
            history_attended,
            current,
            history_weight,
            current_weight,
            grad_history_logit,
            grad_history_attended,
            grad_current,
            grad_current_logit,
            hidden_size=ctx.hidden_size,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_history_logit, grad_history_attended, grad_current, grad_current_logit


def triton_attres_current_finalize(
    history_logit: torch.Tensor,
    history_attended: torch.Tensor,
    current: torch.Tensor,
    current_key: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not all(value.is_cuda for value in (history_logit, history_attended, current, current_key, query)):
        raise ValueError("triton_attres_current_finalize requires CUDA tensors")
    if (
        history_logit.ndim != 2
        or history_attended.ndim != 3
        or current.ndim != 3
        or current_key.ndim != 3
        or query.ndim != 1
    ):
        raise ValueError(
            "expected history_logit [seq, batch], history/current [seq, batch, hidden], query [hidden]"
        )
    if (
        history_attended.shape != current.shape
        or current_key.shape != current.shape
        or history_logit.shape != current.shape[:-1]
    ):
        raise ValueError("AttnRes current-finalize input shapes do not match")
    if int(query.shape[0]) != int(current.shape[-1]):
        raise ValueError("AttnRes current-finalize query hidden size does not match")
    if any(
        value.dtype != torch.bfloat16
        for value in (history_logit, history_attended, current, current_key, query)
    ):
        raise ValueError("triton_attres_current_finalize currently requires BF16 tensors")
    if not all(value.is_contiguous() for value in (history_attended, current, current_key, query)):
        raise ValueError("triton_attres_current_finalize requires contiguous value/query tensors")
    current_logit = torch.matmul(current_key, query)
    return _TritonAttresCurrentFinalize.apply(
        history_logit,
        history_attended,
        current,
        current_logit,
    )


class _TritonAttresWeightedSum(Function):
    @staticmethod
    def forward(ctx, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if not _TRITON_AVAILABLE or not weights.is_cuda or not values.is_cuda:
            raise RuntimeError("Triton AttnRes weighted sum requires CUDA tensors")
        if weights.dtype != values.dtype or weights.dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError(
                "Triton AttnRes weighted sum requires matching bfloat16 or float32 tensors, "
                f"got weights={weights.dtype} values={values.dtype}"
            )
        if weights.ndim != 3 or values.ndim != 4:
            raise ValueError("expected weights [history, seq, batch] and values [history, seq, batch, hidden]")
        if not weights.is_contiguous() or not values.is_contiguous():
            raise ValueError("AttnRes weighted-sum inputs must be contiguous")

        history_size, seq_len, batch_size = weights.shape
        if values.shape[:3] != (history_size, seq_len, batch_size):
            raise ValueError("AttnRes weights and values shapes do not match")
        hidden_size = values.shape[-1]
        seq_batch = seq_len * batch_size
        attended = torch.empty((seq_len, batch_size, hidden_size), dtype=values.dtype, device=values.device)
        block_h = _pow2_block(
            _env_int("MEGATRON_ATTRES_TRITON_WEIGHTED_SUM_BLOCK_H", 512),
            default=512,
            minimum=32,
            maximum=512,
        )
        num_warps = _env_int("MEGATRON_ATTRES_TRITON_WEIGHTED_SUM_NUM_WARPS", 4)
        if history_size == 8:
            attended = torch.einsum("nsb,nsbh->sbh", weights, values)
        else:
            _attres_weighted_sum_forward_kernel[(seq_batch,)](
                weights,
                values,
                attended,
                seq_batch=seq_batch,
                history_size=history_size,
                hidden_size=hidden_size,
                block_h=block_h,
                num_warps=num_warps,
            )
        ctx.save_for_backward(weights, values)
        ctx.shape = (history_size, seq_len, batch_size, hidden_size)
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        return attended

    @staticmethod
    def backward(ctx, grad_attended: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        weights, values = ctx.saved_tensors
        history_size, seq_len, batch_size, hidden_size = ctx.shape
        if grad_attended is None:
            grad_attended = torch.zeros(
                (seq_len, batch_size, hidden_size),
                dtype=values.dtype,
                device=values.device,
            )
        else:
            grad_attended = grad_attended.contiguous()

        # Keep the hidden reduction in PyTorch so dweights follows the exact
        # reduction order used by the unfused AttnRes implementation.
        if history_size == 1:
            grad_weights = (grad_attended.unsqueeze(0) * values).sum(dim=-1)
        else:
            grad_weights = torch.einsum("sbh,nsbh->nsb", grad_attended, values)
        grad_values = torch.empty_like(values)
        seq_batch = seq_len * batch_size
        _attres_weighted_sum_values_backward_kernel[(history_size * seq_batch,)](
            weights,
            grad_attended,
            grad_values,
            seq_batch=seq_batch,
            hidden_size=hidden_size,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_weights, grad_values


def triton_attres_weighted_sum(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return _TritonAttresWeightedSum.apply(weights, values)


def paired_attres_history_logits(keys: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Precompute two history-query products without building an autograd graph."""
    if keys.ndim != 4 or queries.ndim != 2 or int(queries.shape[0]) != 2:
        raise ValueError("expected keys [history, seq, batch, hidden] and queries [2, hidden]")
    if int(keys.shape[-1]) != int(queries.shape[-1]):
        raise ValueError("AttnRes history keys and paired queries do not match")
    with torch.no_grad():
        return torch.einsum("rh,nsbh->rnsb", queries, keys).contiguous()


class _ReuseAttresHistoryLogits(Function):
    @staticmethod
    def forward(ctx, keys: torch.Tensor, query: torch.Tensor, cached_logits: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 4 or query.ndim != 1 or cached_logits.ndim != 3:
            raise ValueError("expected keys [history, seq, batch, hidden], query [hidden], logits [history, seq, batch]")
        if tuple(keys.shape[:-1]) != tuple(cached_logits.shape) or int(keys.shape[-1]) != int(query.shape[0]):
            raise ValueError("AttnRes cached history-logit shapes do not match")
        ctx.save_for_backward(keys, query)
        return cached_logits

    @staticmethod
    def backward(ctx, grad_logits: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        keys, query = ctx.saved_tensors
        if grad_logits is None:
            return None, None, None
        grad_logits = grad_logits.contiguous()
        grad_keys = grad_logits.unsqueeze(-1) * query if ctx.needs_input_grad[0] else None
        grad_query = None
        if ctx.needs_input_grad[1]:
            hidden_size = int(keys.shape[-1])
            grad_query = torch.matmul(keys.reshape(-1, hidden_size).T, grad_logits.reshape(-1))
        return grad_keys, grad_query, None


def reuse_attres_history_logits(
    keys: torch.Tensor,
    query: torch.Tensor,
    cached_logits: torch.Tensor,
) -> torch.Tensor:
    return _ReuseAttresHistoryLogits.apply(keys, query, cached_logits)
