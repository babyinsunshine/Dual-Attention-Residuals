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


_ARG_DEBUG_PRINTED = False


def _debug_arg(name: str, value) -> str:
    value_type = type(value).__name__
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None:
        return f"{name}:type={value_type},shape={tuple(shape)},dtype={dtype},device={device}"
    return f"{name}:type={value_type},value={value!r}"


def _maybe_print_launch_debug(kernel_name: str, grid, args: list[tuple[str, object]]) -> None:
    global _ARG_DEBUG_PRINTED
    if _ARG_DEBUG_PRINTED:
        return
    enabled = os.environ.get("MEGATRON_DUAL_TRITON_FUSED_RESIDUAL_READ_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return
    _ARG_DEBUG_PRINTED = True
    print(f"[dual_triton_arg_debug] kernel={kernel_name} grid={grid!r} grid_type={type(grid).__name__}", flush=True)
    for arg_name, arg_value in args:
        print(f"[dual_triton_arg_debug] {_debug_arg(arg_name, arg_value)}", flush=True)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _pow2_block(value: int, *, default: int, minimum: int = 16, maximum: int = 1024) -> int:
    value = int(value)
    if value <= 0:
        value = int(default)
    value = max(int(minimum), min(int(maximum), value))
    # Triton block dimensions used by tl.arange are expected to be powers of two.
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
    def _te_attention_delta_kernel(
        attention_input,
        raw_attention,
        delta_out,
        elements: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < elements
        hidden = tl.load(attention_input + offsets, mask=mask, other=0.0).to(tl.float32)
        raw = tl.load(raw_attention + offsets, mask=mask, other=0.0).to(tl.float32)
        with_residual = _round_bf16_fp32(hidden + raw)
        delta = _round_bf16_fp32(with_residual - hidden)
        tl.store(delta_out + offsets, delta, mask=mask)


    @triton.jit
    def _dual_te_attention_epilogue_kernel(
        attention_input,
        raw_attention,
        residual0,
        residual1,
        beta,
        delta_out,
        merged_out,
        hidden_size: tl.constexpr,
        rows: tl.constexpr,
        has_residual: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.program_id(1) * block_h + tl.arange(0, block_h)
        mask = (row < rows) & (h < hidden_size)
        offset = row * hidden_size + h
        hidden = tl.load(attention_input + offset, mask=mask, other=0.0).to(tl.float32)
        raw = tl.load(raw_attention + offset, mask=mask, other=0.0).to(tl.float32)
        with_residual = _round_bf16_fp32(hidden + raw)
        delta = _round_bf16_fp32(with_residual - hidden)
        beta0 = tl.load(beta + row * 2 + 0).to(tl.float32)
        beta1 = tl.load(beta + row * 2 + 1).to(tl.float32)
        if has_residual:
            value0 = tl.load(residual0 + offset, mask=mask, other=0.0).to(tl.float32)
            value1 = tl.load(residual1 + offset, mask=mask, other=0.0).to(tl.float32)
            merged0 = tl.fma(delta, beta0, value0)
            merged1 = tl.fma(delta, beta1, value1)
        else:
            merged0 = _round_bf16_fp32(delta * beta0)
            merged1 = _round_bf16_fp32(delta * beta1)
        merged_base = row * 2 * hidden_size
        tl.store(delta_out + offset, delta, mask=mask)
        tl.store(merged_out + merged_base + h, merged0, mask=mask)
        tl.store(merged_out + merged_base + hidden_size + h, merged1, mask=mask)


    @triton.jit
    def _dual_deferred_merge_forward_kernel(
        stream0,
        stream1,
        delta,
        beta,
        keep,
        merged_out,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.program_id(1) * block_h + tl.arange(0, block_h)
        mask = h < hidden_size
        offset = row * hidden_size + h
        value0 = tl.load(stream0 + offset, mask=mask, other=0.0).to(tl.float32)
        value1 = tl.load(stream1 + offset, mask=mask, other=0.0).to(tl.float32)
        delta_value = tl.load(delta + offset, mask=mask, other=0.0).to(tl.float32)
        keep_value = tl.load(keep + row).to(tl.float32)
        beta0 = tl.load(beta + row * 2).to(tl.float32)
        beta1 = tl.load(beta + row * 2 + 1).to(tl.float32)

        mixed0 = _round_bf16_fp32(tl.fma(keep_value, value0 - value1, value1))
        mixed1 = _round_bf16_fp32(tl.fma(keep_value, value1 - value0, value0))
        merged0 = tl.fma(delta_value, beta0, mixed0)
        merged1 = tl.fma(delta_value, beta1, mixed1)
        output_base = row * 2 * hidden_size
        tl.store(merged_out + output_base + h, merged0, mask=mask)
        tl.store(merged_out + output_base + hidden_size + h, merged1, mask=mask)


    @triton.jit
    def _dual_deferred_merge_backward_kernel(
        grad_merged,
        beta,
        keep,
        grad_stream0,
        grad_stream1,
        grad_delta,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.program_id(1) * block_h + tl.arange(0, block_h)
        mask = h < hidden_size
        input_base = row * hidden_size
        output_base = row * 2 * hidden_size
        keep_value = tl.load(keep + row).to(tl.float32)
        one_minus_keep = _round_bf16_fp32(1.0 - keep_value)
        beta0 = tl.load(beta + row * 2).to(tl.float32)
        beta1 = tl.load(beta + row * 2 + 1).to(tl.float32)
        grad0 = tl.load(grad_merged + output_base + h, mask=mask, other=0.0).to(tl.float32)
        grad1 = tl.load(
            grad_merged + output_base + hidden_size + h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        stream0_grad = _round_bf16_fp32(
            _round_bf16_fp32(grad0 * keep_value)
            + _round_bf16_fp32(grad1 * one_minus_keep)
        )
        stream1_grad = _round_bf16_fp32(
            _round_bf16_fp32(grad1 * keep_value)
            + _round_bf16_fp32(grad0 * one_minus_keep)
        )
        delta_grad = _round_bf16_fp32(
            _round_bf16_fp32(grad0 * beta0) + _round_bf16_fp32(grad1 * beta1)
        )
        tl.store(grad_stream0 + input_base + h, stream0_grad, mask=mask)
        tl.store(grad_stream1 + input_base + h, stream1_grad, mask=mask)
        tl.store(grad_delta + input_base + h, delta_grad, mask=mask)

    @triton.jit
    def _fixed_rms_norm_forward_kernel(
        x,
        output,
        inv_rms_out,
        hidden_size: tl.constexpr,
        eps,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        row_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            tl.store(output + row_base + h, value * inv_rms, mask=mask)
        tl.store(inv_rms_out + row, inv_rms)


    @triton.jit
    def _fixed_rms_norm_recompute_kernel(
        x,
        output,
        hidden_size: tl.constexpr,
        eps,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        row_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            tl.store(output + row_base + h, value * inv_rms, mask=mask)


    @triton.jit
    def _fixed_rms_inv_kernel(
        x,
        inv_rms_out,
        hidden_size: tl.constexpr,
        eps,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        row_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        tl.store(inv_rms_out + row, inv_rms)


    @triton.jit
    def _dual_history_key_pack_norm_kernel(
        packed_blocks,
        keys_out,
        history_size: tl.constexpr,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        eps,
        cross_key: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        stream = row % 2
        q = row // 2
        batch = q % batch_size
        q = q // batch_size
        seq = q % seq_len
        history = q // seq_len
        offsets = tl.arange(0, block_h)
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        input_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(packed_blocks + input_base + h, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        output_stream = 1 - stream if cross_key else stream
        output_row = ((output_stream * history_size + history) * batch_size + batch) * seq_len + seq
        output_base = output_row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(packed_blocks + input_base + h, mask=mask, other=0.0).to(tl.float32)
            tl.store(keys_out + output_base + h, value * inv_rms, mask=mask)


    @triton.jit
    def _dual_history_append_prefix_copy_kernel(
        prev_keys,
        prev_values,
        keys_out,
        values_out,
        prefix_elements: tl.constexpr,
        output_history_size: tl.constexpr,
        history_size: tl.constexpr,
        rows_per_history: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < prefix_elements
        row = offsets // rows_per_history
        row_inner = offsets - row * rows_per_history
        stream = row // history_size
        history = row - stream * history_size
        output_offsets = (stream * output_history_size + history) * rows_per_history + row_inner
        key = tl.load(prev_keys + offsets, mask=mask, other=0.0)
        value = tl.load(prev_values + offsets, mask=mask, other=0.0)
        tl.store(keys_out + output_offsets, key, mask=mask)
        tl.store(values_out + output_offsets, value, mask=mask)


    @triton.jit
    def _dual_history_append_new_forward_kernel(
        block,
        keys_out,
        values_out,
        inv_rms_out,
        output_history_size: tl.constexpr,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        eps,
        cross_key: tl.constexpr,
        cross_value: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        input_stream = row % 2
        q = row // 2
        batch = q % batch_size
        seq = q // batch_size
        offsets = tl.arange(0, block_h)
        input_base = ((seq * batch_size + batch) * 2 + input_stream) * hidden_size
        sum_sq = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(block + input_base + h, mask=mask, other=0.0).to(tl.float32)
            sum_sq += value * value
        inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=0) / hidden_size + eps)
        key_stream = 1 - input_stream if cross_key else input_stream
        value_stream = 1 - input_stream if cross_value else input_stream
        key_base = (
            ((key_stream * output_history_size + output_history_size - 1) * batch_size + batch)
            * seq_len
            + seq
        ) * hidden_size
        value_base = (
            ((value_stream * output_history_size + output_history_size - 1) * batch_size + batch)
            * seq_len
            + seq
        ) * hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(block + input_base + h, mask=mask, other=0.0).to(tl.float32)
            tl.store(keys_out + key_base + h, value * inv_rms, mask=mask)
            tl.store(values_out + value_base + h, value, mask=mask)
        tl.store(inv_rms_out + row, inv_rms)


    @triton.jit
    def _dual_history_append_new_backward_kernel(
        grad_keys,
        grad_values,
        block,
        inv_rms,
        grad_block,
        output_history_size: tl.constexpr,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        cross_key: tl.constexpr,
        cross_value: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        input_stream = row % 2
        q = row // 2
        batch = q % batch_size
        seq = q // batch_size
        key_stream = 1 - input_stream if cross_key else input_stream
        value_stream = 1 - input_stream if cross_value else input_stream
        key_base = (
            ((key_stream * output_history_size + output_history_size - 1) * batch_size + batch)
            * seq_len
            + seq
        ) * hidden_size
        value_base = (
            ((value_stream * output_history_size + output_history_size - 1) * batch_size + batch)
            * seq_len
            + seq
        ) * hidden_size
        input_base = ((seq * batch_size + batch) * 2 + input_stream) * hidden_size
        offsets = tl.arange(0, block_h)
        dot = tl.zeros((block_h,), dtype=tl.float32)
        inv = tl.load(inv_rms + row)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(block + input_base + h, mask=mask, other=0.0).to(tl.float32)
            grad_key = tl.load(grad_keys + key_base + h, mask=mask, other=0.0).to(tl.float32)
            dot += grad_key * value
        correction = tl.sum(dot, axis=0) * inv * inv * inv / hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(block + input_base + h, mask=mask, other=0.0).to(tl.float32)
            grad_key = tl.load(grad_keys + key_base + h, mask=mask, other=0.0).to(tl.float32)
            grad_value = tl.load(grad_values + value_base + h, mask=mask, other=0.0).to(tl.float32)
            grad_from_key = _round_bf16_fp32(grad_key * inv - value * correction)
            grad = _round_bf16_fp32(grad_value + grad_from_key)
            tl.store(grad_block + input_base + h, grad, mask=mask)


    @triton.jit
    def _fixed_rms_norm_backward_kernel(
        grad_output,
        x,
        inv_rms,
        grad_x,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        row_base = row * hidden_size
        grad_dot = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(grad_output + row_base + h, mask=mask, other=0.0).to(tl.float32)
            grad_dot += grad * value
        inv = tl.load(inv_rms + row)
        correction = tl.sum(grad_dot, axis=0) * inv * inv * inv / hidden_size
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            value = tl.load(x + row_base + h, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(grad_output + row_base + h, mask=mask, other=0.0).to(tl.float32)
            tl.store(grad_x + row_base + h, grad * inv - value * correction, mask=mask)


    @triton.jit
    def _fused_prefix_residual_params_kernel(
        history_logit,
        history_attended,
        current_streams,
        current_key_streams,
        query,
        fused_weight,
        static_alpha,
        static_beta,
        static_residual_param,
        pre_branch_scale,
        h_post_scale,
        residual_scale,
        param_buf,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        flat_size: tl.constexpr,
        eps,
        cross_key: tl.constexpr,
        bf16_parity: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        seq_idx = row // batch_size
        batch_idx = row - seq_idx * batch_size
        offsets = tl.arange(0, block_h)

        dot_q0 = tl.zeros((block_h,), dtype=tl.float32)
        dot_q1 = tl.zeros((block_h,), dtype=tl.float32)
        stream_base = ((seq_idx * batch_size + batch_idx) * 2) * hidden_size
        key_stream0 = 1 if cross_key else 0
        key_stream1 = 0 if cross_key else 1
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            k0 = tl.load(
                current_key_streams + stream_base + key_stream0 * hidden_size + h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            k1 = tl.load(
                current_key_streams + stream_base + key_stream1 * hidden_size + h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            q0 = tl.load(query + h, mask=mask, other=0.0).to(tl.float32)
            q1 = tl.load(query + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
            dot_q0 += k0 * q0
            dot_q1 += k1 * q1

        current_logit0 = tl.sum(dot_q0, axis=0)
        current_logit1 = tl.sum(dot_q1, axis=0)
        if bf16_parity:
            current_logit0 = current_logit0.to(tl.bfloat16).to(tl.float32)
            current_logit1 = current_logit1.to(tl.bfloat16).to(tl.float32)
        hist_logit0 = tl.load(history_logit + batch_idx * seq_len + seq_idx).to(tl.float32)
        hist_logit1 = tl.load(history_logit + batch_size * seq_len + batch_idx * seq_len + seq_idx).to(tl.float32)
        history_weight0 = tl.sigmoid(hist_logit0 - current_logit0)
        history_weight1 = tl.sigmoid(hist_logit1 - current_logit1)
        current_weight0 = 1.0 - history_weight0
        current_weight1 = 1.0 - history_weight1
        if bf16_parity:
            history_weight0 = history_weight0.to(tl.bfloat16).to(tl.float32)
            history_weight1 = history_weight1.to(tl.bfloat16).to(tl.float32)
            current_weight0 = tl.sigmoid(current_logit0 - hist_logit0).to(tl.bfloat16).to(tl.float32)
            current_weight1 = tl.sigmoid(current_logit1 - hist_logit1).to(tl.bfloat16).to(tl.float32)

        sum_att = tl.zeros((block_h,), dtype=tl.float32)
        sum_partial = tl.zeros((block_h,), dtype=tl.float32)
        acc_ab0 = tl.zeros((block_h,), dtype=tl.float32)
        acc_ab1 = tl.zeros((block_h,), dtype=tl.float32)
        acc_ab2 = tl.zeros((block_h,), dtype=tl.float32)
        acc_ab3 = tl.zeros((block_h,), dtype=tl.float32)
        acc_keep = tl.zeros((block_h,), dtype=tl.float32)

        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            p0 = tl.load(current_streams + stream_base + h, mask=mask, other=0.0).to(tl.float32)
            p1 = tl.load(current_streams + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
            ha0 = tl.load(history_attended + stream_base + h, mask=mask, other=0.0).to(tl.float32)
            ha1 = tl.load(history_attended + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
            a0 = ha0 * history_weight0 + p0 * current_weight0
            a1 = ha1 * history_weight1 + p1 * current_weight1
            if bf16_parity:
                history_part0 = (ha0 * history_weight0).to(tl.bfloat16)
                history_part1 = (ha1 * history_weight1).to(tl.bfloat16)
                current_part0 = (p0 * current_weight0).to(tl.bfloat16)
                current_part1 = (p1 * current_weight1).to(tl.bfloat16)
                a0 = (history_part0 + current_part0).to(tl.bfloat16).to(tl.float32)
                a1 = (history_part1 + current_part1).to(tl.bfloat16).to(tl.float32)
            sum_att += a0 * a0 + a1 * a1
            sum_partial += p0 * p0 + p1 * p1

            f0 = h
            f1 = hidden_size + h
            w0_0 = tl.load(fused_weight + f0 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
            w0_1 = tl.load(fused_weight + f0 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
            w0_2 = tl.load(fused_weight + f0 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
            w0_3 = tl.load(fused_weight + f0 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
            w0_k = tl.load(fused_weight + f0 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
            w1_0 = tl.load(fused_weight + f1 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
            w1_1 = tl.load(fused_weight + f1 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
            w1_2 = tl.load(fused_weight + f1 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
            w1_3 = tl.load(fused_weight + f1 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
            w1_k = tl.load(fused_weight + f1 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
            acc_ab0 += a0 * w0_0 + a1 * w1_0
            acc_ab1 += a0 * w0_1 + a1 * w1_1
            acc_ab2 += a0 * w0_2 + a1 * w1_2
            acc_ab3 += a0 * w0_3 + a1 * w1_3
            acc_keep += p0 * w0_k + p1 * w1_k

        inv_att = 1.0 / tl.sqrt(tl.sum(sum_att, axis=0) / flat_size + eps)
        inv_partial = 1.0 / tl.sqrt(tl.sum(sum_partial, axis=0) / flat_size + eps)
        pre_scale = tl.load(pre_branch_scale).to(tl.float32)
        post_scale = tl.load(h_post_scale).to(tl.float32)
        res_scale = tl.load(residual_scale).to(tl.float32)
        z_ab0 = tl.sum(acc_ab0, axis=0) * inv_att
        z_ab1 = tl.sum(acc_ab1, axis=0) * inv_att
        z_ab2 = tl.sum(acc_ab2, axis=0) * inv_att
        z_ab3 = tl.sum(acc_ab3, axis=0) * inv_att
        z_keep = tl.sum(acc_keep, axis=0) * inv_partial
        static_alpha0 = tl.load(static_alpha + 0).to(tl.float32)
        static_alpha1 = tl.load(static_alpha + 1).to(tl.float32)
        static_beta0 = tl.load(static_beta + 0).to(tl.float32)
        static_beta1 = tl.load(static_beta + 1).to(tl.float32)
        static_residual = tl.load(static_residual_param).to(tl.float32)
        if bf16_parity:
            z_ab0 = z_ab0.to(tl.bfloat16).to(tl.float32)
            z_ab1 = z_ab1.to(tl.bfloat16).to(tl.float32)
            z_ab2 = z_ab2.to(tl.bfloat16).to(tl.float32)
            z_ab3 = z_ab3.to(tl.bfloat16).to(tl.float32)
            z_keep = z_keep.to(tl.bfloat16).to(tl.float32)
            alpha_logit0 = ((z_ab0 * pre_scale).to(tl.bfloat16) + static_alpha0).to(tl.bfloat16)
            alpha_logit1 = ((z_ab1 * pre_scale).to(tl.bfloat16) + static_alpha1).to(tl.bfloat16)
            beta_logit0 = ((z_ab2 * post_scale).to(tl.bfloat16) + static_beta0).to(tl.bfloat16)
            beta_logit1 = ((z_ab3 * post_scale).to(tl.bfloat16) + static_beta1).to(tl.bfloat16)
            keep_logit = ((z_keep * res_scale).to(tl.bfloat16) + static_residual).to(tl.bfloat16)
            alpha0 = tl.sigmoid(alpha_logit0.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            alpha1 = tl.sigmoid(alpha_logit1.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            beta0 = (tl.sigmoid(beta_logit0.to(tl.float32)).to(tl.bfloat16) * 2.0).to(tl.bfloat16).to(tl.float32)
            beta1 = (tl.sigmoid(beta_logit1.to(tl.float32)).to(tl.bfloat16) * 2.0).to(tl.bfloat16).to(tl.float32)
            keep = tl.sigmoid(keep_logit.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        else:
            alpha0 = tl.sigmoid(z_ab0 * pre_scale + static_alpha0)
            alpha1 = tl.sigmoid(z_ab1 * pre_scale + static_alpha1)
            beta0 = tl.sigmoid(z_ab2 * post_scale + static_beta0) * 2.0
            beta1 = tl.sigmoid(z_ab3 * post_scale + static_beta1) * 2.0
            keep = tl.sigmoid(z_keep * res_scale + static_residual)

        out_base = row * 16
        tl.store(param_buf + out_base + 0, alpha0)
        tl.store(param_buf + out_base + 1, alpha1)
        tl.store(param_buf + out_base + 2, beta0)
        tl.store(param_buf + out_base + 3, beta1)
        tl.store(param_buf + out_base + 4, keep)
        tl.store(param_buf + out_base + 5, history_weight0)
        tl.store(param_buf + out_base + 6, history_weight1)
        tl.store(param_buf + out_base + 7, current_weight0)
        tl.store(param_buf + out_base + 8, current_weight1)
        tl.store(param_buf + out_base + 9, z_ab0)
        tl.store(param_buf + out_base + 10, z_ab1)
        tl.store(param_buf + out_base + 11, z_ab2)
        tl.store(param_buf + out_base + 12, z_ab3)
        tl.store(param_buf + out_base + 13, z_keep)
        tl.store(param_buf + out_base + 14, inv_att)
        tl.store(param_buf + out_base + 15, inv_partial)


    @triton.jit
    def _fused_prefix_residual_apply_kernel(
        history_attended,
        current_streams,
        param_buf,
        branch_out,
        residual0_out,
        residual1_out,
        beta_out,
        keep_out,
        hidden_size: tl.constexpr,
        bf16_parity: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        h = block * block_h + tl.arange(0, block_h)
        mask = h < hidden_size
        stream_base = row * 2 * hidden_size

        param_base = row * 16
        alpha0 = tl.load(param_buf + param_base + 0).to(tl.float32)
        alpha1 = tl.load(param_buf + param_base + 1).to(tl.float32)
        beta0 = tl.load(param_buf + param_base + 2).to(tl.float32)
        beta1 = tl.load(param_buf + param_base + 3).to(tl.float32)
        keep = tl.load(param_buf + param_base + 4).to(tl.float32)
        hw0 = tl.load(param_buf + param_base + 5).to(tl.float32)
        hw1 = tl.load(param_buf + param_base + 6).to(tl.float32)
        cw0 = tl.load(param_buf + param_base + 7).to(tl.float32)
        cw1 = tl.load(param_buf + param_base + 8).to(tl.float32)

        p0 = tl.load(current_streams + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        p1 = tl.load(current_streams + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        ha0 = tl.load(history_attended + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        ha1 = tl.load(history_attended + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        a0 = ha0 * hw0 + p0 * cw0
        a1 = ha1 * hw1 + p1 * cw1
        if bf16_parity:
            a0 = ((ha0 * hw0).to(tl.bfloat16) + (p0 * cw0).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
            a1 = ((ha1 * hw1).to(tl.bfloat16) + (p1 * cw1).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
        branch = alpha0 * a0 + alpha1 * a1
        if bf16_parity:
            branch = ((alpha0 * a0).to(tl.bfloat16) + (alpha1 * a1).to(tl.bfloat16)).to(tl.bfloat16)
        residual0 = p1 + keep * (p0 - p1)
        residual1 = p0 + keep * (p1 - p0)

        tl.store(branch_out + row * hidden_size + h, branch, mask=mask)
        tl.store(residual0_out + row * hidden_size + h, residual0, mask=mask)
        tl.store(residual1_out + row * hidden_size + h, residual1, mask=mask)
        first_block = block == 0
        tl.store(beta_out + row * 2 + 0, beta0, mask=first_block)
        tl.store(beta_out + row * 2 + 1, beta1, mask=first_block)
        tl.store(keep_out + row, keep, mask=first_block)


    @triton.jit
    def _fused_prefix_residual_post_backward_kernel(
        history_attended,
        current_streams,
        grad_branch,
        grad_residual0,
        grad_residual1,
        fused_weight,
        grad_dot_ab,
        grad_inv_attended,
        inv_attended,
        grad_dot_keep,
        grad_inv_current,
        inv_current,
        history_weight,
        current_weight,
        alpha,
        keep,
        grad_history_attended_out,
        grad_current_out,
        partial_sums,
        hidden_size: tl.constexpr,
        flat_size: tl.constexpr,
        num_blocks: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        h = block * block_h + tl.arange(0, block_h)
        mask = h < hidden_size
        stream_base = row * 2 * hidden_size

        weight_base = row * 2
        h0 = tl.load(history_weight + weight_base + 0).to(tl.float32)
        h1 = tl.load(history_weight + weight_base + 1).to(tl.float32)
        c0 = tl.load(current_weight + weight_base + 0).to(tl.float32)
        c1 = tl.load(current_weight + weight_base + 1).to(tl.float32)
        a0_scale = tl.load(alpha + weight_base + 0).to(tl.float32)
        a1_scale = tl.load(alpha + weight_base + 1).to(tl.float32)
        keep_value = tl.load(keep + row).to(tl.float32)
        one_minus_keep = 1.0 - keep_value

        gb = tl.load(grad_branch + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        gr0 = tl.load(grad_residual0 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        gr1 = tl.load(grad_residual1 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        ha0 = tl.load(history_attended + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        ha1 = tl.load(history_attended + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        cur0 = tl.load(current_streams + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        cur1 = tl.load(current_streams + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        attended0 = ha0 * h0 + cur0 * c0
        attended1 = ha1 * h1 + cur1 * c1

        gdab0 = tl.load(grad_dot_ab + row * 4 + 0).to(tl.float32)
        gdab1 = tl.load(grad_dot_ab + row * 4 + 1).to(tl.float32)
        gdab2 = tl.load(grad_dot_ab + row * 4 + 2).to(tl.float32)
        gdab3 = tl.load(grad_dot_ab + row * 4 + 3).to(tl.float32)
        gdk = tl.load(grad_dot_keep + row).to(tl.float32)
        inv_att = tl.load(inv_attended + row).to(tl.float32)
        inv_cur = tl.load(inv_current + row).to(tl.float32)
        norm_att = tl.load(grad_inv_attended + row).to(tl.float32) * inv_att * inv_att * inv_att / flat_size
        norm_cur = tl.load(grad_inv_current + row).to(tl.float32) * inv_cur * inv_cur * inv_cur / flat_size

        f0 = h
        f1 = hidden_size + h
        fw0_0 = tl.load(fused_weight + f0 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
        fw0_1 = tl.load(fused_weight + f0 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
        fw0_2 = tl.load(fused_weight + f0 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
        fw0_3 = tl.load(fused_weight + f0 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
        fw0_4 = tl.load(fused_weight + f0 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
        fw1_0 = tl.load(fused_weight + f1 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
        fw1_1 = tl.load(fused_weight + f1 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
        fw1_2 = tl.load(fused_weight + f1 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
        fw1_3 = tl.load(fused_weight + f1 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
        fw1_4 = tl.load(fused_weight + f1 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
        gfa0 = gdab0 * fw0_0 + gdab1 * fw0_1 + gdab2 * fw0_2 + gdab3 * fw0_3 - norm_att * attended0
        gfa1 = gdab0 * fw1_0 + gdab1 * fw1_1 + gdab2 * fw1_2 + gdab3 * fw1_3 - norm_att * attended1
        gfc0 = gdk * fw0_4 - norm_cur * cur0
        gfc1 = gdk * fw1_4 - norm_cur * cur1

        grad_att0 = gb * a0_scale + gfa0
        grad_att1 = gb * a1_scale + gfa1
        grad_cur0 = gr0 * keep_value + gr1 * one_minus_keep + gfc0 + grad_att0 * c0
        grad_cur1 = gr0 * one_minus_keep + gr1 * keep_value + gfc1 + grad_att1 * c1
        grad_hist_att0 = grad_att0 * h0
        grad_hist_att1 = grad_att1 * h1

        tl.store(grad_history_attended_out + stream_base + h, grad_hist_att0, mask=mask)
        tl.store(grad_history_attended_out + stream_base + hidden_size + h, grad_hist_att1, mask=mask)
        tl.store(grad_current_out + stream_base + h, grad_cur0, mask=mask)
        tl.store(grad_current_out + stream_base + hidden_size + h, grad_cur1, mask=mask)

        partial_base = (row * num_blocks + block) * 4
        tl.store(partial_sums + partial_base + 0, tl.sum(grad_att0 * ha0, axis=0))
        tl.store(partial_sums + partial_base + 1, tl.sum(grad_att1 * ha1, axis=0))
        tl.store(partial_sums + partial_base + 2, tl.sum(grad_att0 * cur0, axis=0))
        tl.store(partial_sums + partial_base + 3, tl.sum(grad_att1 * cur1, axis=0))


    @triton.jit
    def _fused_prefix_residual_key_query_backward_kernel(
        current_key_streams,
        query,
        grad_current_logits,
        grad_current_key_out,
        grad_query_partials,
        hidden_size: tl.constexpr,
        total_rows: tl.constexpr,
        num_h_blocks: tl.constexpr,
        cross_key: tl.constexpr,
        block_r: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        h_block = tl.program_id(1)
        rows = row_block * block_r + tl.arange(0, block_r)
        h_offsets = tl.arange(0, block_h)
        h = h_block * block_h + h_offsets
        row_mask = rows < total_rows
        h_mask = h < hidden_size
        matrix_mask = row_mask[:, None] & h_mask[None, :]

        key_stream0 = 1 if cross_key else 0
        key_stream1 = 0 if cross_key else 1
        grad_key_stream0 = key_stream0
        grad_key_stream1 = key_stream1

        g0 = tl.load(grad_current_logits + rows * 2 + 0, mask=row_mask, other=0.0).to(tl.float32)
        g1 = tl.load(grad_current_logits + rows * 2 + 1, mask=row_mask, other=0.0).to(tl.float32)
        q0 = tl.load(query + h, mask=h_mask, other=0.0).to(tl.float32)
        q1 = tl.load(query + hidden_size + h, mask=h_mask, other=0.0).to(tl.float32)

        key0_offsets = (rows[:, None] * 2 + key_stream0) * hidden_size + h[None, :]
        key1_offsets = (rows[:, None] * 2 + key_stream1) * hidden_size + h[None, :]
        k0 = tl.load(current_key_streams + key0_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
        k1 = tl.load(current_key_streams + key1_offsets, mask=matrix_mask, other=0.0).to(tl.float32)

        grad_key0 = g0[:, None] * q0[None, :]
        grad_key1 = g1[:, None] * q1[None, :]
        grad_key0_offsets = (rows[:, None] * 2 + grad_key_stream0) * hidden_size + h[None, :]
        grad_key1_offsets = (rows[:, None] * 2 + grad_key_stream1) * hidden_size + h[None, :]
        tl.store(grad_current_key_out + grad_key0_offsets, grad_key0, mask=matrix_mask)
        tl.store(grad_current_key_out + grad_key1_offsets, grad_key1, mask=matrix_mask)

        partial_base = (row_block * 2 * num_h_blocks + h_block) * block_h
        tl.store(grad_query_partials + partial_base + h_offsets, tl.sum(g0[:, None] * k0, axis=0), mask=h_mask)
        tl.store(
            grad_query_partials + partial_base + num_h_blocks * block_h + h_offsets,
            tl.sum(g1[:, None] * k1, axis=0),
            mask=h_mask,
        )


    @triton.jit
    def _fused_prefix_residual_full_row_backward_kernel(
        history_attended,
        current_streams,
        grad_branch,
        grad_residual0,
        grad_residual1,
        grad_beta,
        grad_keep_out,
        param_buf,
        pre_branch_scale,
        h_post_scale,
        residual_scale,
        row_grads,
        hidden_size: tl.constexpr,
        flat_size: tl.constexpr,
        has_grad_branch: tl.constexpr,
        has_grad_residual0: tl.constexpr,
        has_grad_residual1: tl.constexpr,
        has_grad_beta: tl.constexpr,
        has_grad_keep: tl.constexpr,
        bf16_parity: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        stream_base = row * 2 * hidden_size
        param_base = row * 16

        h0 = tl.load(param_buf + param_base + 5)
        h1 = tl.load(param_buf + param_base + 6)
        c0 = tl.load(param_buf + param_base + 7)
        c1 = tl.load(param_buf + param_base + 8)
        alpha0 = tl.load(param_buf + param_base + 0)
        alpha1 = tl.load(param_buf + param_base + 1)
        beta0 = tl.load(param_buf + param_base + 2)
        beta1 = tl.load(param_buf + param_base + 3)
        keep = tl.load(param_buf + param_base + 4)
        z0 = tl.load(param_buf + param_base + 9)
        z1 = tl.load(param_buf + param_base + 10)
        z2 = tl.load(param_buf + param_base + 11)
        z3 = tl.load(param_buf + param_base + 12)
        z_keep = tl.load(param_buf + param_base + 13)
        inv_att = tl.load(param_buf + param_base + 14)
        inv_cur = tl.load(param_buf + param_base + 15)

        grad_alpha0_acc = tl.zeros((block_h,), dtype=tl.float32)
        grad_alpha1_acc = tl.zeros((block_h,), dtype=tl.float32)
        grad_keep_acc = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            h = block_start + offsets
            mask = h < hidden_size
            cur0 = tl.load(current_streams + stream_base + h, mask=mask, other=0.0).to(tl.float32)
            cur1 = tl.load(current_streams + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
            ha0 = tl.load(history_attended + stream_base + h, mask=mask, other=0.0).to(tl.float32)
            ha1 = tl.load(history_attended + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
            attended0 = ha0 * h0 + cur0 * c0
            attended1 = ha1 * h1 + cur1 * c1
            if bf16_parity:
                attended0 = ((ha0 * h0).to(tl.bfloat16) + (cur0 * c0).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
                attended1 = ((ha1 * h1).to(tl.bfloat16) + (cur1 * c1).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
            if has_grad_branch:
                gb = tl.load(grad_branch + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
                grad_alpha0_acc += gb * attended0
                grad_alpha1_acc += gb * attended1
            if has_grad_residual0:
                gr0 = tl.load(grad_residual0 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
                grad_keep_acc += gr0 * (cur0 - cur1)
            if has_grad_residual1:
                gr1 = tl.load(grad_residual1 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
                grad_keep_acc += gr1 * (cur1 - cur0)

        grad_alpha0 = tl.sum(grad_alpha0_acc, axis=0)
        grad_alpha1 = tl.sum(grad_alpha1_acc, axis=0)
        grad_beta0 = 0.0
        grad_beta1 = 0.0
        if has_grad_beta:
            grad_beta0 = tl.load(grad_beta + row * 2 + 0).to(tl.float32)
            grad_beta1 = tl.load(grad_beta + row * 2 + 1).to(tl.float32)
        grad_keep = tl.sum(grad_keep_acc, axis=0)
        if has_grad_keep:
            grad_keep += tl.load(grad_keep_out + row).to(tl.float32)

        grad_y_alpha0 = grad_alpha0 * alpha0 * (1.0 - alpha0)
        grad_y_alpha1 = grad_alpha1 * alpha1 * (1.0 - alpha1)
        beta_sigmoid0 = beta0 * 0.5
        beta_sigmoid1 = beta1 * 0.5
        grad_y_beta0 = grad_beta0 * (2.0 * beta_sigmoid0 * (1.0 - beta_sigmoid0))
        grad_y_beta1 = grad_beta1 * (2.0 * beta_sigmoid1 * (1.0 - beta_sigmoid1))
        grad_y_keep = grad_keep * keep * (1.0 - keep)
        pre_scale = tl.load(pre_branch_scale).to(tl.float32)
        post_scale = tl.load(h_post_scale).to(tl.float32)
        res_scale = tl.load(residual_scale).to(tl.float32)
        grad_z0 = grad_y_alpha0 * pre_scale
        grad_z1 = grad_y_alpha1 * pre_scale
        grad_z2 = grad_y_beta0 * post_scale
        grad_z3 = grad_y_beta1 * post_scale
        grad_z_keep = grad_y_keep * res_scale

        row_base = row * 15
        tl.store(row_grads + row_base + 0, grad_z0 * inv_att)
        tl.store(row_grads + row_base + 1, grad_z1 * inv_att)
        tl.store(row_grads + row_base + 2, grad_z2 * inv_att)
        tl.store(row_grads + row_base + 3, grad_z3 * inv_att)
        # Match the reference backward's FP32 operation order.  Reassociating
        # these terms changes a few ulps that deep residual stacks can amplify.
        grad_inv_att = (
            grad_z0 * (z0 / inv_att)
            + grad_z1 * (z1 / inv_att)
            + grad_z2 * (z2 / inv_att)
            + grad_z3 * (z3 / inv_att)
        )
        tl.store(
            row_grads + row_base + 4,
            grad_inv_att * inv_att * inv_att * inv_att / flat_size,
        )
        tl.store(row_grads + row_base + 5, grad_z_keep * inv_cur)
        grad_inv_cur = grad_z_keep * (z_keep / inv_cur)
        tl.store(
            row_grads + row_base + 6,
            grad_inv_cur * inv_cur * inv_cur * inv_cur / flat_size,
        )
        tl.store(row_grads + row_base + 7, grad_y_alpha0)
        tl.store(row_grads + row_base + 8, grad_y_alpha1)
        tl.store(row_grads + row_base + 9, grad_y_beta0)
        tl.store(row_grads + row_base + 10, grad_y_beta1)
        tl.store(row_grads + row_base + 11, grad_y_keep)
        tl.store(row_grads + row_base + 12, grad_y_alpha0 * z0 + grad_y_alpha1 * z1)
        tl.store(row_grads + row_base + 13, grad_y_beta0 * z2 + grad_y_beta1 * z3)
        tl.store(row_grads + row_base + 14, grad_y_keep * z_keep)


    @triton.jit
    def _fused_prefix_residual_full_post_backward_kernel(
        history_attended,
        current_streams,
        grad_branch,
        grad_residual0,
        grad_residual1,
        fused_weight,
        param_buf,
        row_grads,
        grad_history_attended_out,
        grad_current_out,
        partial_sums,
        hidden_size: tl.constexpr,
        num_blocks: tl.constexpr,
        has_grad_branch: tl.constexpr,
        has_grad_residual0: tl.constexpr,
        has_grad_residual1: tl.constexpr,
        bf16_parity: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        h = block * block_h + tl.arange(0, block_h)
        mask = h < hidden_size
        stream_base = row * 2 * hidden_size
        param_base = row * 16
        row_base = row * 15

        h0 = tl.load(param_buf + param_base + 5)
        h1 = tl.load(param_buf + param_base + 6)
        c0 = tl.load(param_buf + param_base + 7)
        c1 = tl.load(param_buf + param_base + 8)
        alpha0 = tl.load(param_buf + param_base + 0)
        alpha1 = tl.load(param_buf + param_base + 1)
        keep = tl.load(param_buf + param_base + 4)
        one_minus_keep = 1.0 - keep
        grad_dot0 = tl.load(row_grads + row_base + 0)
        grad_dot1 = tl.load(row_grads + row_base + 1)
        grad_dot2 = tl.load(row_grads + row_base + 2)
        grad_dot3 = tl.load(row_grads + row_base + 3)
        norm_att = tl.load(row_grads + row_base + 4)
        grad_dot_keep = tl.load(row_grads + row_base + 5)
        norm_cur = tl.load(row_grads + row_base + 6)

        gb = tl.zeros((block_h,), dtype=tl.float32)
        gr0 = tl.zeros((block_h,), dtype=tl.float32)
        gr1 = tl.zeros((block_h,), dtype=tl.float32)
        if has_grad_branch:
            gb = tl.load(grad_branch + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        if has_grad_residual0:
            gr0 = tl.load(grad_residual0 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        if has_grad_residual1:
            gr1 = tl.load(grad_residual1 + row * hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        ha0 = tl.load(history_attended + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        ha1 = tl.load(history_attended + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        cur0 = tl.load(current_streams + stream_base + h, mask=mask, other=0.0).to(tl.float32)
        cur1 = tl.load(current_streams + stream_base + hidden_size + h, mask=mask, other=0.0).to(tl.float32)
        attended0 = ha0 * h0 + cur0 * c0
        attended1 = ha1 * h1 + cur1 * c1
        if bf16_parity:
            attended0 = ((ha0 * h0).to(tl.bfloat16) + (cur0 * c0).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
            attended1 = ((ha1 * h1).to(tl.bfloat16) + (cur1 * c1).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)

        f0 = h
        f1 = hidden_size + h
        fw0_0 = tl.load(fused_weight + f0 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
        fw0_1 = tl.load(fused_weight + f0 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
        fw0_2 = tl.load(fused_weight + f0 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
        fw0_3 = tl.load(fused_weight + f0 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
        fw0_4 = tl.load(fused_weight + f0 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
        fw1_0 = tl.load(fused_weight + f1 * 5 + 0, mask=mask, other=0.0).to(tl.float32)
        fw1_1 = tl.load(fused_weight + f1 * 5 + 1, mask=mask, other=0.0).to(tl.float32)
        fw1_2 = tl.load(fused_weight + f1 * 5 + 2, mask=mask, other=0.0).to(tl.float32)
        fw1_3 = tl.load(fused_weight + f1 * 5 + 3, mask=mask, other=0.0).to(tl.float32)
        fw1_4 = tl.load(fused_weight + f1 * 5 + 4, mask=mask, other=0.0).to(tl.float32)
        grad_flat_att0 = (
            grad_dot0 * fw0_0 + grad_dot1 * fw0_1 + grad_dot2 * fw0_2 + grad_dot3 * fw0_3
            - norm_att * attended0
        )
        grad_flat_att1 = (
            grad_dot0 * fw1_0 + grad_dot1 * fw1_1 + grad_dot2 * fw1_2 + grad_dot3 * fw1_3
            - norm_att * attended1
        )
        grad_flat_cur0 = grad_dot_keep * fw0_4 - norm_cur * cur0
        grad_flat_cur1 = grad_dot_keep * fw1_4 - norm_cur * cur1
        grad_att0 = gb * alpha0 + grad_flat_att0
        grad_att1 = gb * alpha1 + grad_flat_att1
        grad_cur0 = gr0 * keep + gr1 * one_minus_keep + grad_flat_cur0 + grad_att0 * c0
        grad_cur1 = gr0 * one_minus_keep + gr1 * keep + grad_flat_cur1 + grad_att1 * c1

        tl.store(grad_history_attended_out + stream_base + h, grad_att0 * h0, mask=mask)
        tl.store(grad_history_attended_out + stream_base + hidden_size + h, grad_att1 * h1, mask=mask)
        tl.store(grad_current_out + stream_base + h, grad_cur0, mask=mask)
        tl.store(grad_current_out + stream_base + hidden_size + h, grad_cur1, mask=mask)
        partial_base = (row * num_blocks + block) * 4
        tl.store(partial_sums + partial_base + 0, tl.sum(grad_att0 * ha0, axis=0))
        tl.store(partial_sums + partial_base + 1, tl.sum(grad_att1 * ha1, axis=0))
        tl.store(partial_sums + partial_base + 2, tl.sum(grad_att0 * cur0, axis=0))
        tl.store(partial_sums + partial_base + 3, tl.sum(grad_att1 * cur1, axis=0))


    @triton.jit
    def _fused_prefix_residual_full_finalize_backward_kernel(
        partial_sums,
        param_buf,
        grad_history_logit_out,
        grad_current_logits_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        num_blocks: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        blocks = tl.arange(0, block_n)
        block_mask = blocks < num_blocks
        partial_base = (row * num_blocks + blocks) * 4
        grad_history0 = tl.sum(
            tl.load(partial_sums + partial_base + 0, mask=block_mask, other=0.0),
            axis=0,
        )
        grad_history1 = tl.sum(
            tl.load(partial_sums + partial_base + 1, mask=block_mask, other=0.0),
            axis=0,
        )
        grad_current0 = tl.sum(
            tl.load(partial_sums + partial_base + 2, mask=block_mask, other=0.0),
            axis=0,
        )
        grad_current1 = tl.sum(
            tl.load(partial_sums + partial_base + 3, mask=block_mask, other=0.0),
            axis=0,
        )
        param_base = row * 16
        history_weight0 = tl.load(param_buf + param_base + 5)
        history_weight1 = tl.load(param_buf + param_base + 6)
        current_weight0 = tl.load(param_buf + param_base + 7)
        current_weight1 = tl.load(param_buf + param_base + 8)
        grad_hist0 = (
            grad_history0 * history_weight0 * (1.0 - history_weight0)
            - grad_current0 * current_weight0 * (1.0 - current_weight0)
        )
        grad_hist1 = (
            grad_history1 * history_weight1 * (1.0 - history_weight1)
            - grad_current1 * current_weight1 * (1.0 - current_weight1)
        )
        seq_idx = row // batch_size
        batch_idx = row - seq_idx * batch_size
        history_offset = batch_idx * seq_len + seq_idx
        tl.store(grad_history_logit_out + history_offset, grad_hist0)
        tl.store(grad_history_logit_out + batch_size * seq_len + history_offset, grad_hist1)
        tl.store(grad_current_logits_out + row * 2 + 0, -grad_hist0)
        tl.store(grad_current_logits_out + row * 2 + 1, -grad_hist1)


    @triton.jit
    def _fused_prefix_residual_full_scalar_backward_kernel(
        row_grads,
        scalar_grads_out,
        total_rows: tl.constexpr,
        block_r: tl.constexpr,
    ):
        column = tl.program_id(0)
        rows = tl.arange(0, block_r)
        mask = rows < total_rows
        values = tl.load(row_grads + rows * 15 + 7 + column, mask=mask, other=0.0)
        tl.store(scalar_grads_out + column, tl.sum(values, axis=0))


    @triton.jit
    def _fused_prefix_residual_full_weight_backward_kernel(
        history_attended,
        current_streams,
        param_buf,
        row_grads,
        grad_fused_out,
        total_rows: tl.constexpr,
        hidden_size: tl.constexpr,
        bf16_parity: tl.constexpr,
        block_r: tl.constexpr,
        block_f: tl.constexpr,
    ):
        features = tl.program_id(0) * block_f + tl.arange(0, block_f)
        feature_mask = features < 2 * hidden_size
        stream = features // hidden_size
        h = features - stream * hidden_size
        rows_offset = tl.arange(0, block_r)
        acc0 = tl.zeros((block_f,), dtype=tl.float32)
        acc1 = tl.zeros((block_f,), dtype=tl.float32)
        acc2 = tl.zeros((block_f,), dtype=tl.float32)
        acc3 = tl.zeros((block_f,), dtype=tl.float32)
        acc4 = tl.zeros((block_f,), dtype=tl.float32)
        for row_start in tl.range(0, total_rows, block_r):
            rows = row_start + rows_offset
            row_mask = rows < total_rows
            matrix_mask = row_mask[:, None] & feature_mask[None, :]
            value_offset = (rows[:, None] * 2 + stream[None, :]) * hidden_size + h[None, :]
            cur = tl.load(current_streams + value_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            hist = tl.load(history_attended + value_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            weight = tl.load(
                param_buf + rows[:, None] * 16 + 5 + stream[None, :],
                mask=matrix_mask,
                other=0.0,
            )
            current_weight = tl.load(
                param_buf + rows[:, None] * 16 + 7 + stream[None, :],
                mask=matrix_mask,
                other=0.0,
            )
            attended = hist * weight + cur * current_weight
            if bf16_parity:
                attended = (
                    (hist * weight).to(tl.bfloat16) + (cur * current_weight).to(tl.bfloat16)
                ).to(tl.bfloat16).to(tl.float32)
            grad_base = rows * 15
            grad0 = tl.load(row_grads + grad_base + 0, mask=row_mask, other=0.0)
            grad1 = tl.load(row_grads + grad_base + 1, mask=row_mask, other=0.0)
            grad2 = tl.load(row_grads + grad_base + 2, mask=row_mask, other=0.0)
            grad3 = tl.load(row_grads + grad_base + 3, mask=row_mask, other=0.0)
            grad4 = tl.load(row_grads + grad_base + 5, mask=row_mask, other=0.0)
            acc0 += tl.sum(attended * grad0[:, None], axis=0)
            acc1 += tl.sum(attended * grad1[:, None], axis=0)
            acc2 += tl.sum(attended * grad2[:, None], axis=0)
            acc3 += tl.sum(attended * grad3[:, None], axis=0)
            acc4 += tl.sum(cur * grad4[:, None], axis=0)
        output_base = features * 5
        tl.store(grad_fused_out + output_base + 0, acc0, mask=feature_mask)
        tl.store(grad_fused_out + output_base + 1, acc1, mask=feature_mask)
        tl.store(grad_fused_out + output_base + 2, acc2, mask=feature_mask)
        tl.store(grad_fused_out + output_base + 3, acc3, mask=feature_mask)
        tl.store(grad_fused_out + output_base + 4, acc4, mask=feature_mask)


    @triton.jit
    def _fused_prefix_residual_full_key_query_backward_kernel(
        current_key_streams,
        current_streams,
        current_inv_rms,
        query,
        grad_current_logits,
        grad_current_key_out,
        grad_query_out,
        hidden_size: tl.constexpr,
        total_rows: tl.constexpr,
        cross_key: tl.constexpr,
        recompute_key: tl.constexpr,
        block_r: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        h_block = tl.program_id(1)
        rows = row_block * block_r + tl.arange(0, block_r)
        h = h_block * block_h + tl.arange(0, block_h)
        row_mask = rows < total_rows
        h_mask = h < hidden_size
        matrix_mask = row_mask[:, None] & h_mask[None, :]
        key_stream0 = 1 if cross_key else 0
        key_stream1 = 0 if cross_key else 1
        grad0 = tl.load(grad_current_logits + rows * 2 + 0, mask=row_mask, other=0.0)
        grad1 = tl.load(grad_current_logits + rows * 2 + 1, mask=row_mask, other=0.0)
        query0 = tl.load(query + h, mask=h_mask, other=0.0).to(tl.float32)
        query1 = tl.load(query + hidden_size + h, mask=h_mask, other=0.0).to(tl.float32)
        key0_offset = (rows[:, None] * 2 + key_stream0) * hidden_size + h[None, :]
        key1_offset = (rows[:, None] * 2 + key_stream1) * hidden_size + h[None, :]
        if recompute_key:
            raw0 = tl.load(current_streams + key0_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            raw1 = tl.load(current_streams + key1_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            inv0 = tl.load(current_inv_rms + rows * 2 + key_stream0, mask=row_mask, other=0.0)
            inv1 = tl.load(current_inv_rms + rows * 2 + key_stream1, mask=row_mask, other=0.0)
            key0 = (raw0 * inv0[:, None]).to(tl.bfloat16).to(tl.float32)
            key1 = (raw1 * inv1[:, None]).to(tl.bfloat16).to(tl.float32)
        else:
            key0 = tl.load(current_key_streams + key0_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            key1 = tl.load(current_key_streams + key1_offset, mask=matrix_mask, other=0.0).to(tl.float32)
        tl.store(grad_current_key_out + key0_offset, grad0[:, None] * query0[None, :], mask=matrix_mask)
        tl.store(grad_current_key_out + key1_offset, grad1[:, None] * query1[None, :], mask=matrix_mask)
        query_grad0 = tl.sum(grad0[:, None] * key0, axis=0)
        query_grad1 = tl.sum(grad1[:, None] * key1, axis=0)
        tl.atomic_add(grad_query_out + h, query_grad0, mask=h_mask)
        tl.atomic_add(grad_query_out + hidden_size + h, query_grad1, mask=h_mask)


    @triton.jit
    def _history_prefix_softmax_forward_kernel(
        logits,
        scores_out,
        denom_out,
        history_logit_out,
        history_size: tl.constexpr,
        batch_seq: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        rd = row // batch_seq
        bs = row - rd * batch_seq
        n = tl.arange(0, block_n)
        mask = n < history_size
        offsets = rd * history_size * batch_seq + n * batch_seq + bs

        logits_f32 = tl.load(logits + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        history_max = tl.max(logits_f32, axis=0)
        centered = (logits_f32 - history_max).to(tl.bfloat16).to(tl.float32)
        scores = tl.exp(centered).to(tl.bfloat16)
        denom = tl.sum(scores.to(tl.float32), axis=0).to(tl.bfloat16)
        log_denom = tl.log(denom.to(tl.float32)).to(tl.bfloat16)
        history_logit = (history_max + log_denom.to(tl.float32)).to(tl.bfloat16)

        tl.store(scores_out + offsets, scores, mask=mask)
        tl.store(denom_out + row, denom)
        tl.store(history_logit_out + row, history_logit)


    @triton.jit
    def _history_prefix_softmax_backward_kernel(
        scores,
        denom,
        grad_scores,
        grad_denom,
        grad_history_logit,
        grad_logits_out,
        history_size: tl.constexpr,
        batch_seq: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        rd = row // batch_seq
        bs = row - rd * batch_seq
        n = tl.arange(0, block_n)
        mask = n < history_size
        offsets = rd * history_size * batch_seq + n * batch_seq + bs

        score = tl.load(scores + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_score = tl.load(grad_scores + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_denom_value = tl.load(grad_denom + row).to(tl.float32)
        grad_logit = tl.load(grad_history_logit + row).to(tl.float32)
        denom_value = tl.load(denom + row).to(tl.float32)

        grad_log_denom = (grad_logit / denom_value).to(tl.bfloat16).to(tl.float32)
        grad_denom_total = (grad_denom_value + grad_log_denom).to(tl.bfloat16).to(tl.float32)
        grad_score_total = (grad_score + grad_denom_total).to(tl.bfloat16).to(tl.float32)
        grad_centered = (grad_score_total * score).to(tl.bfloat16).to(tl.float32)
        grad_centered_sum = tl.sum(grad_centered, axis=0).to(tl.bfloat16).to(tl.float32)
        grad_max = (grad_logit - grad_centered_sum).to(tl.bfloat16).to(tl.float32)
        max_index = tl.argmax(score, axis=0)
        grad_logits = grad_centered + tl.where(n == max_index, grad_max, 0.0)
        tl.store(grad_logits_out + offsets, grad_logits.to(tl.bfloat16), mask=mask)


    @triton.jit
    def _history_prefix_normalize_forward_kernel(
        numerator,
        denom,
        attended_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        depth: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, block_h)
        r = row // (seq_len * batch_size * depth)
        sbd = row - r * seq_len * batch_size * depth
        seq_idx = sbd // (batch_size * depth)
        bd = sbd - seq_idx * batch_size * depth
        batch_idx = bd // depth
        depth_idx = bd - batch_idx * depth
        denom_offset = ((r * depth + depth_idx) * batch_size + batch_idx) * seq_len + seq_idx
        denom_value = tl.load(denom + denom_offset).to(tl.float32)
        row_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            mask = hidden < hidden_size
            value = tl.load(numerator + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            tl.store(attended_out + row_base + hidden, value / denom_value, mask=mask)


    @triton.jit
    def _history_prefix_normalize_backward_kernel(
        grad_attended,
        numerator,
        denom,
        grad_numerator_out,
        grad_denom_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        depth: tl.constexpr,
        hidden_size: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, block_h)
        r = row // (seq_len * batch_size * depth)
        sbd = row - r * seq_len * batch_size * depth
        seq_idx = sbd // (batch_size * depth)
        bd = sbd - seq_idx * batch_size * depth
        batch_idx = bd // depth
        depth_idx = bd - batch_idx * depth
        denom_offset = ((r * depth + depth_idx) * batch_size + batch_idx) * seq_len + seq_idx
        denom_value = tl.load(denom + denom_offset).to(tl.float32)
        row_base = row * hidden_size
        grad_denom_acc = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            mask = hidden < hidden_size
            grad = tl.load(grad_attended + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            value = tl.load(numerator + row_base + hidden, mask=mask, other=0.0).to(tl.float32)
            tl.store(grad_numerator_out + row_base + hidden, grad / denom_value, mask=mask)
            first_div = (value / denom_value).to(tl.bfloat16).to(tl.float32)
            second_div = (first_div / denom_value).to(tl.bfloat16).to(tl.float32)
            grad_denom_element = ((-grad) * second_div).to(tl.bfloat16).to(tl.float32)
            grad_denom_acc += tl.where(mask, grad_denom_element, 0.0)
        tl.store(grad_denom_out + denom_offset, tl.sum(grad_denom_acc, axis=0))


    @triton.jit
    def _history_prefix_pair_forward_kernel(
        logits,
        values,
        scores_out,
        denom_out,
        history_logit_out,
        attended_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        depth: tl.constexpr,
        history_size: tl.constexpr,
        hidden_size: tl.constexpr,
        batch_seq: tl.constexpr,
        values_seq_major: tl.constexpr,
        block_n: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        depth_idx = row // batch_seq
        bs = row - depth_idx * batch_seq
        batch_idx = bs // seq_len
        seq_idx = bs - batch_idx * seq_len
        n = tl.arange(0, block_n)
        n_mask = n < history_size

        logits_base0 = depth_idx * history_size * batch_seq + bs
        logits_base1 = (depth + depth_idx) * history_size * batch_seq + bs
        logits_offsets0 = logits_base0 + n * batch_seq
        logits_offsets1 = logits_base1 + n * batch_seq
        logits0 = tl.load(logits + logits_offsets0, mask=n_mask, other=-float("inf")).to(tl.float32)
        logits1 = tl.load(logits + logits_offsets1, mask=n_mask, other=-float("inf")).to(tl.float32)
        history_max0 = tl.max(logits0, axis=0)
        history_max1 = tl.max(logits1, axis=0)
        centered0 = (logits0 - history_max0).to(tl.bfloat16).to(tl.float32)
        centered1 = (logits1 - history_max1).to(tl.bfloat16).to(tl.float32)
        scores0 = tl.exp(centered0).to(tl.bfloat16)
        scores1 = tl.exp(centered1).to(tl.bfloat16)
        denom0 = tl.sum(scores0.to(tl.float32), axis=0).to(tl.bfloat16)
        denom1 = tl.sum(scores1.to(tl.float32), axis=0).to(tl.bfloat16)
        log_denom0 = tl.log(denom0.to(tl.float32)).to(tl.bfloat16)
        log_denom1 = tl.log(denom1.to(tl.float32)).to(tl.bfloat16)
        history_logit0 = (history_max0 + log_denom0.to(tl.float32)).to(tl.bfloat16)
        history_logit1 = (history_max1 + log_denom1.to(tl.float32)).to(tl.bfloat16)

        tl.store(scores_out + logits_offsets0, scores0, mask=n_mask)
        tl.store(scores_out + logits_offsets1, scores1, mask=n_mask)
        tl.store(denom_out + row, denom0)
        tl.store(denom_out + depth * batch_seq + row, denom1)
        tl.store(history_logit_out + row, history_logit0)
        tl.store(history_logit_out + depth * batch_seq + row, history_logit1)

        h = tl.arange(0, block_h)
        value_bs = seq_idx * batch_size + batch_idx if values_seq_major else bs
        values_base = (depth_idx * history_size * batch_seq + value_bs) * hidden_size
        output_base0 = ((seq_idx * batch_size + batch_idx) * depth + depth_idx) * hidden_size
        output_base1 = output_base0 + seq_len * batch_size * depth * hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            hidden_mask = hidden < hidden_size
            numerator0 = tl.zeros((block_h,), dtype=tl.float32)
            numerator1 = tl.zeros((block_h,), dtype=tl.float32)
            for history_idx in range(0, history_size):
                value_offsets = values_base + history_idx * batch_seq * hidden_size + hidden
                value = tl.load(values + value_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
                score0 = tl.sum(tl.where(n == history_idx, scores0.to(tl.float32), 0.0), axis=0)
                score1 = tl.sum(tl.where(n == history_idx, scores1.to(tl.float32), 0.0), axis=0)
                numerator0 += score0 * value
                numerator1 += score1 * value
            numerator0 = numerator0.to(tl.bfloat16)
            numerator1 = numerator1.to(tl.bfloat16)
            attended0 = numerator0.to(tl.float32) / denom0.to(tl.float32)
            attended1 = numerator1.to(tl.float32) / denom1.to(tl.float32)
            tl.store(attended_out + output_base0 + hidden, attended0, mask=hidden_mask)
            tl.store(attended_out + output_base1 + hidden, attended1, mask=hidden_mask)


    @triton.jit
    def _history_prefix_pair_logits_backward_kernel(
        logits,
        scores,
        values,
        denom,
        attended,
        grad_history_logit,
        grad_attended,
        grad_logits_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        depth: tl.constexpr,
        history_size: tl.constexpr,
        hidden_size: tl.constexpr,
        batch_seq: tl.constexpr,
        values_seq_major: tl.constexpr,
        block_n: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        depth_idx = row // batch_seq
        bs = row - depth_idx * batch_seq
        batch_idx = bs // seq_len
        seq_idx = bs - batch_idx * seq_len
        n = tl.arange(0, block_n)
        n_mask = n < history_size
        score_base0 = depth_idx * history_size * batch_seq + bs
        score_base1 = (depth + depth_idx) * history_size * batch_seq + bs
        score_offsets0 = score_base0 + n * batch_seq
        score_offsets1 = score_base1 + n * batch_seq
        logit0 = tl.load(logits + score_offsets0, mask=n_mask, other=-float("inf")).to(tl.float32)
        logit1 = tl.load(logits + score_offsets1, mask=n_mask, other=-float("inf")).to(tl.float32)
        score0 = tl.load(scores + score_offsets0, mask=n_mask, other=0.0).to(tl.float32)
        score1 = tl.load(scores + score_offsets1, mask=n_mask, other=0.0).to(tl.float32)
        denom0 = tl.load(denom + row).to(tl.float32)
        denom1 = tl.load(denom + depth * batch_seq + row).to(tl.float32)

        h = tl.arange(0, block_h)
        value_bs = seq_idx * batch_size + batch_idx if values_seq_major else bs
        values_base = (depth_idx * history_size * batch_seq + value_bs) * hidden_size
        output_base0 = ((seq_idx * batch_size + batch_idx) * depth + depth_idx) * hidden_size
        output_base1 = output_base0 + seq_len * batch_size * depth * hidden_size
        grad_score0 = tl.zeros((block_n,), dtype=tl.float32)
        grad_score1 = tl.zeros((block_n,), dtype=tl.float32)
        grad_denom_elements0 = tl.zeros((block_h,), dtype=tl.float32)
        grad_denom_elements1 = tl.zeros((block_h,), dtype=tl.float32)
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            hidden_mask = hidden < hidden_size
            grad0 = tl.load(grad_attended + output_base0 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            grad1 = tl.load(grad_attended + output_base1 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            attended0 = tl.load(attended + output_base0 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            attended1 = tl.load(attended + output_base1 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            grad_num0 = (grad0 / denom0).to(tl.bfloat16).to(tl.float32)
            grad_num1 = (grad1 / denom1).to(tl.bfloat16).to(tl.float32)
            second_div0 = (attended0 / denom0).to(tl.bfloat16).to(tl.float32)
            second_div1 = (attended1 / denom1).to(tl.bfloat16).to(tl.float32)
            grad_denom_elements0 += tl.where(
                hidden_mask,
                ((-grad0) * second_div0).to(tl.bfloat16).to(tl.float32),
                0.0,
            )
            grad_denom_elements1 += tl.where(
                hidden_mask,
                ((-grad1) * second_div1).to(tl.bfloat16).to(tl.float32),
                0.0,
            )
            value_offsets = values_base + n[:, None] * batch_seq * hidden_size + hidden[None, :]
            value_mask = n_mask[:, None] & hidden_mask[None, :]
            value = tl.load(values + value_offsets, mask=value_mask, other=0.0).to(tl.float32)
            grad_score0 += tl.sum(value * grad_num0[None, :], axis=1)
            grad_score1 += tl.sum(value * grad_num1[None, :], axis=1)

        grad_score0 = grad_score0.to(tl.bfloat16).to(tl.float32)
        grad_score1 = grad_score1.to(tl.bfloat16).to(tl.float32)
        grad_denom0 = tl.sum(grad_denom_elements0, axis=0).to(tl.bfloat16).to(tl.float32)
        grad_denom1 = tl.sum(grad_denom_elements1, axis=0).to(tl.bfloat16).to(tl.float32)
        grad_logit0 = tl.load(grad_history_logit + row).to(tl.float32)
        grad_logit1 = tl.load(grad_history_logit + depth * batch_seq + row).to(tl.float32)

        grad_log_denom0 = (grad_logit0 / denom0).to(tl.bfloat16).to(tl.float32)
        grad_log_denom1 = (grad_logit1 / denom1).to(tl.bfloat16).to(tl.float32)
        grad_denom_total0 = (grad_denom0 + grad_log_denom0).to(tl.bfloat16).to(tl.float32)
        grad_denom_total1 = (grad_denom1 + grad_log_denom1).to(tl.bfloat16).to(tl.float32)
        grad_score_total0 = (grad_score0 + grad_denom_total0).to(tl.bfloat16).to(tl.float32)
        grad_score_total1 = (grad_score1 + grad_denom_total1).to(tl.bfloat16).to(tl.float32)
        grad_centered0 = (grad_score_total0 * score0).to(tl.bfloat16).to(tl.float32)
        grad_centered1 = (grad_score_total1 * score1).to(tl.bfloat16).to(tl.float32)
        grad_centered_sum0 = tl.sum(grad_centered0, axis=0).to(tl.bfloat16).to(tl.float32)
        grad_centered_sum1 = tl.sum(grad_centered1, axis=0).to(tl.bfloat16).to(tl.float32)
        grad_max0 = (grad_logit0 - grad_centered_sum0).to(tl.bfloat16).to(tl.float32)
        grad_max1 = (grad_logit1 - grad_centered_sum1).to(tl.bfloat16).to(tl.float32)
        # Scores are stored in BF16 and two nearby logits can therefore map to
        # the same exp value. Use the original logits so max backward follows
        # the same history index as torch.max.
        max_index0 = tl.argmax(logit0, axis=0)
        max_index1 = tl.argmax(logit1, axis=0)
        grad_logits0 = grad_centered0 + tl.where(n == max_index0, grad_max0, 0.0)
        grad_logits1 = grad_centered1 + tl.where(n == max_index1, grad_max1, 0.0)
        tl.store(grad_logits_out + score_offsets0, grad_logits0.to(tl.bfloat16), mask=n_mask)
        tl.store(grad_logits_out + score_offsets1, grad_logits1.to(tl.bfloat16), mask=n_mask)


    @triton.jit
    def _history_prefix_pair_values_backward_kernel(
        scores,
        denom,
        grad_attended,
        grad_values_out,
        seq_len: tl.constexpr,
        batch_size: tl.constexpr,
        depth: tl.constexpr,
        history_size: tl.constexpr,
        hidden_size: tl.constexpr,
        batch_seq: tl.constexpr,
        values_seq_major: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        depth_history = row // batch_seq
        value_bs = row - depth_history * batch_seq
        depth_idx = depth_history // history_size
        history_idx = depth_history - depth_idx * history_size
        if values_seq_major:
            seq_idx = value_bs // batch_size
            batch_idx = value_bs - seq_idx * batch_size
            bs = batch_idx * seq_len + seq_idx
        else:
            bs = value_bs
            batch_idx = bs // seq_len
            seq_idx = bs - batch_idx * seq_len
        score_offset0 = (depth_idx * history_size + history_idx) * batch_seq + bs
        score_offset1 = ((depth + depth_idx) * history_size + history_idx) * batch_seq + bs
        score0 = tl.load(scores + score_offset0).to(tl.float32)
        score1 = tl.load(scores + score_offset1).to(tl.float32)
        denom0 = tl.load(denom + depth_idx * batch_seq + bs).to(tl.float32)
        denom1 = tl.load(denom + (depth + depth_idx) * batch_seq + bs).to(tl.float32)

        h = tl.arange(0, block_h)
        output_base0 = ((seq_idx * batch_size + batch_idx) * depth + depth_idx) * hidden_size
        output_base1 = output_base0 + seq_len * batch_size * depth * hidden_size
        value_base = row * hidden_size
        for block_start in range(0, hidden_size, block_h):
            hidden = block_start + h
            hidden_mask = hidden < hidden_size
            grad0 = tl.load(grad_attended + output_base0 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            grad1 = tl.load(grad_attended + output_base1 + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
            grad_num0 = (grad0 / denom0).to(tl.bfloat16).to(tl.float32)
            grad_num1 = (grad1 / denom1).to(tl.bfloat16).to(tl.float32)
            grad_value = score0 * grad_num0 + score1 * grad_num1
            tl.store(grad_values_out + value_base + hidden, grad_value.to(tl.bfloat16), mask=hidden_mask)


def triton_is_available() -> bool:
    return bool(_TRITON_AVAILABLE)


class _TritonTeAttentionDelta(Function):
    @staticmethod
    def forward(
        ctx,
        attention_input: torch.Tensor,
        raw_attention: torch.Tensor,
        block_h: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE or not attention_input.is_cuda:
            raise RuntimeError("Triton TE-attention delta requires CUDA")
        if attention_input.dtype != torch.bfloat16 or raw_attention.dtype != torch.bfloat16:
            raise TypeError("Triton TE-attention delta currently supports bfloat16 only")
        if attention_input.shape != raw_attention.shape:
            raise ValueError("attention input and raw output shapes must match")
        attention_input = attention_input.contiguous()
        raw_attention = raw_attention.contiguous()
        delta = torch.empty_like(raw_attention)
        elements = int(delta.numel())
        block = _pow2_block(block_h, default=256, minimum=64, maximum=1024)
        _te_attention_delta_kernel[(triton.cdiv(elements, block),)](
            attention_input=attention_input,
            raw_attention=raw_attention,
            delta_out=delta,
            elements=elements,
            block=block,
            num_warps=4,
        )
        return delta

    @staticmethod
    def backward(ctx, grad_delta: torch.Tensor):
        return None, grad_delta, None


def triton_te_attention_delta(
    attention_input: torch.Tensor,
    raw_attention: torch.Tensor,
) -> torch.Tensor:
    return _TritonTeAttentionDelta.apply(
        attention_input,
        raw_attention,
        _env_int("MEGATRON_RESIDUAL_TE_EPILOGUE_BLOCK_H", 1024),
    )


class _TritonDualTeAttentionEpilogue(Function):
    @staticmethod
    def forward(
        ctx,
        attention_input: torch.Tensor,
        raw_attention: torch.Tensor,
        residual0: torch.Tensor,
        residual1: torch.Tensor,
        beta: torch.Tensor,
        has_residual: bool,
        block_h: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE or not attention_input.is_cuda:
            raise RuntimeError("Triton Dual TE-attention epilogue requires CUDA")
        tensors = (attention_input, raw_attention, residual0, residual1, beta)
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise TypeError("Triton Dual TE-attention epilogue currently supports bfloat16 only")
        if attention_input.shape != raw_attention.shape or residual0.shape != attention_input.shape:
            raise ValueError("Dual TE-attention epilogue hidden tensor shapes must match")
        if residual1.shape != attention_input.shape or beta.shape != (*attention_input.shape[:-1], 2):
            raise ValueError("Dual TE-attention epilogue residual/beta shapes do not match")
        attention_input = attention_input.contiguous()
        raw_attention = raw_attention.contiguous()
        residual0 = residual0.contiguous()
        residual1 = residual1.contiguous()
        beta = beta.contiguous()
        hidden_size = int(attention_input.shape[-1])
        rows = int(attention_input.numel() // hidden_size)
        delta = torch.empty_like(raw_attention)
        merged = torch.empty(
            (*attention_input.shape[:-1], 2, hidden_size),
            device=attention_input.device,
            dtype=attention_input.dtype,
        )
        block = _pow2_block(block_h, default=256, minimum=64, maximum=max(64, hidden_size))
        _dual_te_attention_epilogue_kernel[(rows, triton.cdiv(hidden_size, block))](
            attention_input=attention_input,
            raw_attention=raw_attention,
            residual0=residual0,
            residual1=residual1,
            beta=beta,
            delta_out=delta,
            merged_out=merged,
            hidden_size=hidden_size,
            rows=rows,
            has_residual=bool(has_residual),
            block_h=block,
            num_warps=4,
        )
        ctx.has_residual = bool(has_residual)
        ctx.save_for_backward(delta, beta)
        return merged

    @staticmethod
    def backward(ctx, grad_merged: torch.Tensor):
        delta, beta = ctx.saved_tensors
        grad_streams = grad_merged.contiguous()
        grad0 = grad_streams[..., 0, :]
        grad1 = grad_streams[..., 1, :]
        grad_raw = grad0 * beta[..., 0:1] + grad1 * beta[..., 1:2]
        grad_beta = torch.stack(
            (
                (grad0 * delta).sum(dim=-1),
                (grad1 * delta).sum(dim=-1),
            ),
            dim=-1,
        )
        return (
            None,
            grad_raw,
            grad0 if ctx.has_residual else None,
            grad1 if ctx.has_residual else None,
            grad_beta,
            None,
            None,
        )


def triton_dual_te_attention_epilogue(
    attention_input: torch.Tensor,
    raw_attention: torch.Tensor,
    residual0: torch.Tensor,
    residual1: torch.Tensor,
    beta: torch.Tensor,
    *,
    has_residual: bool,
) -> torch.Tensor:
    return _TritonDualTeAttentionEpilogue.apply(
        attention_input,
        raw_attention,
        residual0,
        residual1,
        beta,
        bool(has_residual),
        _env_int("MEGATRON_DUAL_TE_EPILOGUE_BLOCK_H", 512),
    )


class _TritonDualDeferredMerge(Function):
    @staticmethod
    def forward(
        ctx,
        stream0: torch.Tensor,
        stream1: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
        keep: torch.Tensor,
        block_h: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE or not stream0.is_cuda:
            raise RuntimeError("Triton Dual deferred merge requires CUDA")
        tensors = (stream0, stream1, delta, beta, keep)
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise TypeError("Triton Dual deferred merge currently supports bfloat16 only")
        if stream0.shape != stream1.shape or stream0.shape != delta.shape:
            raise ValueError("Dual deferred merge stream and delta shapes must match")
        expected_scalar_shape = stream0.shape[:-1]
        if beta.shape != (*expected_scalar_shape, 2) or keep.shape != expected_scalar_shape:
            raise ValueError("Dual deferred merge beta/keep shapes do not match streams")
        stream0 = stream0.contiguous()
        stream1 = stream1.contiguous()
        delta = delta.contiguous()
        beta = beta.contiguous()
        keep = keep.contiguous()
        hidden_size = int(stream0.shape[-1])
        rows = int(stream0.numel() // hidden_size)
        merged = torch.empty(
            (*expected_scalar_shape, 2, hidden_size),
            device=stream0.device,
            dtype=stream0.dtype,
        )
        block = _pow2_block(block_h, default=256, minimum=64, maximum=max(64, hidden_size))
        _dual_deferred_merge_forward_kernel[(rows, triton.cdiv(hidden_size, block))](
            stream0,
            stream1,
            delta,
            beta,
            keep,
            merged,
            hidden_size=hidden_size,
            block_h=block,
            num_warps=4,
        )
        ctx.block_h = block
        ctx.save_for_backward(stream0, stream1, delta, beta, keep)
        return merged

    @staticmethod
    def backward(ctx, grad_merged: torch.Tensor):
        stream0, stream1, delta, beta, keep = ctx.saved_tensors
        grad_merged = grad_merged.contiguous()
        grad_stream0 = torch.empty_like(stream0)
        grad_stream1 = torch.empty_like(stream1)
        grad_delta = torch.empty_like(delta)
        hidden_size = int(stream0.shape[-1])
        rows = int(stream0.numel() // hidden_size)
        _dual_deferred_merge_backward_kernel[(rows, triton.cdiv(hidden_size, ctx.block_h))](
            grad_merged,
            beta,
            keep,
            grad_stream0,
            grad_stream1,
            grad_delta,
            hidden_size=hidden_size,
            block_h=ctx.block_h,
            num_warps=4,
        )
        grad0 = grad_merged[..., 0, :]
        grad1 = grad_merged[..., 1, :]
        grad_beta = torch.stack(
            ((grad0 * delta).sum(dim=-1), (grad1 * delta).sum(dim=-1)),
            dim=-1,
        )
        stream_difference = stream0 - stream1
        grad_keep = (grad0 * stream_difference).sum(dim=-1) - (grad1 * stream_difference).sum(dim=-1)
        return grad_stream0, grad_stream1, grad_delta, grad_beta, grad_keep, None


def triton_dual_deferred_merge(
    stream0: torch.Tensor,
    stream1: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    return _TritonDualDeferredMerge.apply(
        stream0,
        stream1,
        delta,
        beta,
        keep,
        _env_int("MEGATRON_DUAL_DEFERRED_MERGE_BLOCK_H", 256),
    )


def _as_zeros_like(value: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    return value


class _TritonHistoryPrefixSoftmax(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE or not logits.is_cuda:
            raise RuntimeError("Triton history-prefix softmax requires a CUDA tensor")
        if logits.dtype != torch.bfloat16:
            raise TypeError("Triton history-prefix softmax currently supports bfloat16 only")
        if logits.ndim != 5 or not logits.is_contiguous():
            raise ValueError("history-prefix logits must be contiguous [query, depth, history, batch, seq]")

        queries, depth, history_size, batch_size, seq_len = logits.shape
        batch_seq = batch_size * seq_len
        rows = queries * depth * batch_seq
        scores = torch.empty_like(logits)
        denom = torch.empty(
            (queries, depth, batch_size, seq_len),
            dtype=logits.dtype,
            device=logits.device,
        )
        history_logit = torch.empty_like(denom)
        block_n = triton.next_power_of_2(history_size)
        _history_prefix_softmax_forward_kernel[(rows,)](
            logits,
            scores,
            denom,
            history_logit,
            history_size=history_size,
            batch_seq=batch_seq,
            block_n=block_n,
            num_warps=1,
        )
        ctx.save_for_backward(scores, denom)
        ctx.history_size = history_size
        ctx.batch_seq = batch_seq
        return scores, denom, history_logit

    @staticmethod
    def backward(
        ctx,
        grad_scores: Optional[torch.Tensor],
        grad_denom: Optional[torch.Tensor],
        grad_history_logit: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor]:
        scores, denom = ctx.saved_tensors
        grad_scores = _as_zeros_like(grad_scores, scores).contiguous()
        grad_denom = _as_zeros_like(grad_denom, denom).contiguous()
        grad_history_logit = _as_zeros_like(grad_history_logit, denom).contiguous()
        grad_logits = torch.empty_like(scores)
        rows = denom.numel()
        block_n = triton.next_power_of_2(ctx.history_size)
        _history_prefix_softmax_backward_kernel[(rows,)](
            scores,
            denom,
            grad_scores,
            grad_denom,
            grad_history_logit,
            grad_logits,
            history_size=ctx.history_size,
            batch_seq=ctx.batch_seq,
            block_n=block_n,
            num_warps=1,
        )
        return (grad_logits,)


def triton_history_prefix_softmax(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _TritonHistoryPrefixSoftmax.apply(logits)


class _TritonHistoryPrefixNormalize(Function):
    @staticmethod
    def forward(ctx, numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        if not _TRITON_AVAILABLE or not numerator.is_cuda:
            raise RuntimeError("Triton history-prefix normalization requires CUDA tensors")
        if numerator.dtype != torch.bfloat16 or denom.dtype != torch.bfloat16:
            raise TypeError("Triton history-prefix normalization currently supports bfloat16 only")
        if numerator.ndim != 5 or denom.ndim != 4:
            raise ValueError("expected numerator [query, seq, batch, depth, hidden] and denom [query, depth, batch, seq]")
        if not numerator.is_contiguous() or not denom.is_contiguous():
            raise ValueError("history-prefix numerator and denominator must be contiguous")

        queries, seq_len, batch_size, depth, hidden_size = numerator.shape
        if denom.shape != (queries, depth, batch_size, seq_len):
            raise ValueError("history-prefix numerator and denominator shapes do not match")
        attended = torch.empty_like(numerator)
        rows = queries * seq_len * batch_size * depth
        block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_HISTORY_PREFIX_BLOCK_H", 512),
            default=512,
            minimum=64,
            maximum=1024,
        )
        _history_prefix_normalize_forward_kernel[(rows,)](
            numerator,
            denom,
            attended,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            hidden_size=hidden_size,
            block_h=block_h,
            num_warps=4,
        )
        ctx.save_for_backward(numerator, denom)
        ctx.shape = (queries, seq_len, batch_size, depth, hidden_size)
        ctx.block_h = block_h
        return attended

    @staticmethod
    def backward(
        ctx,
        grad_attended: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numerator, denom = ctx.saved_tensors
        grad_attended = _as_zeros_like(grad_attended, numerator).contiguous()
        grad_numerator = torch.empty_like(numerator)
        grad_denom = torch.empty_like(denom)
        queries, seq_len, batch_size, depth, hidden_size = ctx.shape
        rows = queries * seq_len * batch_size * depth
        _history_prefix_normalize_backward_kernel[(rows,)](
            grad_attended,
            numerator,
            denom,
            grad_numerator,
            grad_denom,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            hidden_size=hidden_size,
            block_h=ctx.block_h,
            num_warps=4,
        )
        return grad_numerator, grad_denom


def triton_history_prefix_normalize(
    numerator: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return _TritonHistoryPrefixNormalize.apply(numerator, denom)


class _TritonHistoryPrefixPair(Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        values: torch.Tensor,
        flip_values: bool,
        *history_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE or not logits.is_cuda or not values.is_cuda:
            raise RuntimeError("Triton history-prefix pair requires CUDA tensors")
        if logits.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise TypeError("Triton history-prefix pair currently supports bfloat16 only")
        if logits.ndim != 5 or values.ndim != 5:
            raise ValueError("expected logits [2, depth, history, batch, seq] and values [depth, history, batch, seq, hidden]")
        if not logits.is_contiguous() or not values.is_contiguous():
            raise ValueError("history-prefix pair inputs must be contiguous")

        queries, depth, history_size, batch_size, seq_len = logits.shape
        if queries != 2:
            raise ValueError(f"history-prefix pair requires exactly two queries, got {queries}")
        if values.shape[:4] != (depth, history_size, batch_size, seq_len):
            raise ValueError("history-prefix pair logits and values shapes do not match")
        hidden_size = values.shape[-1]
        batch_seq = batch_size * seq_len
        scores = torch.empty_like(logits)
        denom = torch.empty(
            (queries, depth, batch_size, seq_len),
            dtype=logits.dtype,
            device=logits.device,
        )
        history_logit = torch.empty_like(denom)
        attended = torch.empty(
            (queries, seq_len, batch_size, depth, hidden_size),
            dtype=values.dtype,
            device=values.device,
        )
        block_n = triton.next_power_of_2(history_size)
        block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_HISTORY_PREFIX_BLOCK_H", 64),
            default=64,
            minimum=32,
            maximum=512,
        )
        num_warps = _env_int("MEGATRON_DUAL_TRITON_HISTORY_PREFIX_NUM_WARPS", 4)
        _history_prefix_pair_forward_kernel[(depth * batch_seq,)](
            logits,
            values,
            scores,
            denom,
            history_logit,
            attended,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=False,
            block_n=block_n,
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.recompute_values = bool(history_blocks)
        ctx.flip_values = bool(flip_values)
        if ctx.recompute_values:
            ctx.save_for_backward(logits, scores, denom, attended, *history_blocks)
        else:
            ctx.save_for_backward(logits, scores, values, denom, attended)
        ctx.shape = (depth, history_size, batch_size, seq_len, hidden_size)
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        return history_logit, attended

    @staticmethod
    def backward(
        ctx,
        grad_history_logit: Optional[torch.Tensor],
        grad_attended: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if ctx.recompute_values:
            logits, scores, denom, attended, *history_blocks = ctx.saved_tensors
            depth, history_size, batch_size, seq_len, hidden_size = ctx.shape
            if len(history_blocks) != history_size:
                raise RuntimeError("Dual history value recompute block count mismatch")
            packed_values = []
            for block in history_blocks:
                streams = block.view(seq_len, batch_size, depth, hidden_size)
                value = streams.transpose(0, 2)
                if ctx.flip_values:
                    value = value.flip(0)
                packed_values.append(value.contiguous())
            values = torch.stack(packed_values, dim=1)
        else:
            logits, scores, values, denom, attended = ctx.saved_tensors
        grad_history_logit = _as_zeros_like(grad_history_logit, denom).contiguous()
        grad_attended = _as_zeros_like(grad_attended, attended).contiguous()
        grad_logits = torch.empty_like(scores)
        grad_values = torch.empty_like(values)
        depth, history_size, batch_size, seq_len, hidden_size = ctx.shape
        batch_seq = batch_size * seq_len
        _history_prefix_pair_logits_backward_kernel[(depth * batch_seq,)](
            logits,
            scores,
            values,
            denom,
            attended,
            grad_history_logit,
            grad_attended,
            grad_logits,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=False,
            block_n=ctx.block_n,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        _history_prefix_pair_values_backward_kernel[(depth * history_size * batch_seq,)](
            scores,
            denom,
            grad_attended,
            grad_values,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=False,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_logits, grad_values, None, *(None for _ in range(history_size if ctx.recompute_values else 0))


def triton_history_prefix_pair(
    logits: torch.Tensor,
    values: torch.Tensor,
    *,
    history_blocks: Optional[list[torch.Tensor]] = None,
    flip_values: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TritonHistoryPrefixPair.apply(logits, values, flip_values, *(history_blocks or ()))


class _TritonAttresHistoryPrefixPair(Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        values: torch.Tensor,
        *history_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE or not logits.is_cuda or not values.is_cuda:
            raise RuntimeError("Triton AttnRes history-prefix pair requires CUDA tensors")
        if logits.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise TypeError("Triton AttnRes history-prefix pair currently supports bfloat16 only")
        if logits.ndim != 4 or values.ndim != 4:
            raise ValueError("expected logits [2, history, batch, seq] and values [history, seq, batch, hidden]")
        if not logits.is_contiguous() or not values.is_contiguous():
            raise ValueError("AttnRes history-prefix pair inputs must be contiguous")

        queries, history_size, batch_size, seq_len = logits.shape
        if queries != 2:
            raise ValueError(f"AttnRes history-prefix pair requires exactly two queries, got {queries}")
        if values.shape[:3] != (history_size, seq_len, batch_size):
            raise ValueError("AttnRes history-prefix pair logits and values shapes do not match")
        depth = 1
        hidden_size = values.shape[-1]
        batch_seq = batch_size * seq_len
        scores = torch.empty_like(logits)
        denom = torch.empty((queries, depth, batch_size, seq_len), dtype=logits.dtype, device=logits.device)
        history_logit = torch.empty_like(denom)
        attended = torch.empty(
            (queries, seq_len, batch_size, depth, hidden_size),
            dtype=values.dtype,
            device=values.device,
        )
        block_n = triton.next_power_of_2(history_size)
        block_h = _pow2_block(
            _env_int("MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_BLOCK_H", 64),
            default=64,
            minimum=32,
            maximum=512,
        )
        num_warps = _env_int("MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_NUM_WARPS", 4)
        _history_prefix_pair_forward_kernel[(batch_seq,)](
            logits,
            values,
            scores,
            denom,
            history_logit,
            attended,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=True,
            block_n=block_n,
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.recompute_values = bool(history_blocks)
        if ctx.recompute_values:
            ctx.save_for_backward(logits, scores, denom, attended, *history_blocks)
        else:
            ctx.save_for_backward(logits, scores, values, denom, attended)
        ctx.shape = (history_size, batch_size, seq_len, hidden_size)
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        return history_logit[:, 0], attended[:, :, :, 0]

    @staticmethod
    def backward(
        ctx,
        grad_history_logit: Optional[torch.Tensor],
        grad_attended: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if ctx.recompute_values:
            logits, scores, denom, attended, *history_blocks = ctx.saved_tensors
            history_size, batch_size, seq_len, hidden_size = ctx.shape
            if len(history_blocks) != history_size:
                raise RuntimeError("AttnRes history value recompute block count mismatch")
            values = torch.stack(tuple(history_blocks), dim=0)
        else:
            logits, scores, values, denom, attended = ctx.saved_tensors
        if grad_history_logit is None:
            grad_history_logit_full = torch.zeros_like(denom)
        else:
            grad_history_logit_full = grad_history_logit.unsqueeze(1).contiguous()
        if grad_attended is None:
            grad_attended_full = torch.zeros_like(attended)
        else:
            grad_attended_full = grad_attended.unsqueeze(3).contiguous()
        grad_logits = torch.empty_like(scores)
        grad_values = torch.empty_like(values)
        history_size, batch_size, seq_len, hidden_size = ctx.shape
        depth = 1
        batch_seq = batch_size * seq_len
        _history_prefix_pair_logits_backward_kernel[(batch_seq,)](
            logits,
            scores,
            values,
            denom,
            attended,
            grad_history_logit_full,
            grad_attended_full,
            grad_logits,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=True,
            block_n=ctx.block_n,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        _history_prefix_pair_values_backward_kernel[(history_size * batch_seq,)](
            scores,
            denom,
            grad_attended_full,
            grad_values,
            seq_len=seq_len,
            batch_size=batch_size,
            depth=depth,
            history_size=history_size,
            hidden_size=hidden_size,
            batch_seq=batch_seq,
            values_seq_major=True,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_logits, grad_values, *(None for _ in range(history_size if ctx.recompute_values else 0))


def triton_attres_history_prefix_pair(
    logits: torch.Tensor,
    values: torch.Tensor,
    *,
    history_blocks: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TritonAttresHistoryPrefixPair.apply(logits, values, *(history_blocks or ()))


def _triton_backward_post_enabled() -> bool:
    return bool(_TRITON_AVAILABLE) and os.environ.get("MEGATRON_DUAL_TRITON_BACKWARD_POST", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _triton_backward_key_query_enabled() -> bool:
    return bool(_TRITON_AVAILABLE) and os.environ.get("MEGATRON_DUAL_TRITON_BACKWARD_KEY_QUERY", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _triton_backward_full_enabled() -> bool:
    return bool(_TRITON_AVAILABLE) and os.environ.get("MEGATRON_DUAL_TRITON_BACKWARD_FULL", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _triton_backward_full_finalize_enabled() -> bool:
    return bool(_TRITON_AVAILABLE) and os.environ.get(
        "MEGATRON_DUAL_TRITON_BACKWARD_FULL_FINALIZE", "1"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _backward_triton_full(
    grad_branch: Optional[torch.Tensor],
    grad_residual0: Optional[torch.Tensor],
    grad_residual1: Optional[torch.Tensor],
    grad_beta: Optional[torch.Tensor],
    grad_keep_out: Optional[torch.Tensor],
    history_logit: torch.Tensor,
    history_attended: torch.Tensor,
    current_streams: torch.Tensor,
    current_key_streams: torch.Tensor,
    query: torch.Tensor,
    fused_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    static_residual_param: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    residual_scale: torch.Tensor,
    param_buf: torch.Tensor,
    norm_eps: float,
    cross_key: bool,
    bf16_parity: bool,
    recompute_current_key: bool = False,
) -> tuple[torch.Tensor, ...]:
    # The kernels use flat contiguous addressing, while autograd may provide
    # expanded or transposed gradient views for residual outputs.
    grad_branch = None if grad_branch is None else grad_branch.contiguous()
    grad_residual0 = (
        None if grad_residual0 is None else grad_residual0.contiguous()
    )
    grad_residual1 = (
        None if grad_residual1 is None else grad_residual1.contiguous()
    )
    grad_beta = None if grad_beta is None else grad_beta.contiguous()
    grad_keep_out = (
        None if grad_keep_out is None else grad_keep_out.contiguous()
    )
    seq_len, batch_size, _, hidden_size = current_streams.shape
    rows = seq_len * batch_size
    flat_size = 2 * hidden_size
    row_block_h = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_ROW_BLOCK_H", 512),
        default=512,
        maximum=max(16, hidden_size),
    )
    post_block_h = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_POST_BLOCK_H", 512),
        default=512,
        maximum=max(16, hidden_size),
    )
    num_post_blocks = triton.cdiv(hidden_size, post_block_h)
    row_grads = torch.empty((rows, 15), device=current_streams.device, dtype=torch.float32)
    dummy = current_streams
    _fused_prefix_residual_full_row_backward_kernel[(rows,)](
        history_attended=history_attended,
        current_streams=current_streams,
        grad_branch=grad_branch if grad_branch is not None else dummy,
        grad_residual0=grad_residual0 if grad_residual0 is not None else dummy,
        grad_residual1=grad_residual1 if grad_residual1 is not None else dummy,
        grad_beta=grad_beta if grad_beta is not None else dummy,
        grad_keep_out=grad_keep_out if grad_keep_out is not None else dummy,
        param_buf=param_buf,
        pre_branch_scale=pre_branch_scale,
        h_post_scale=h_post_scale,
        residual_scale=residual_scale,
        row_grads=row_grads,
        hidden_size=hidden_size,
        flat_size=flat_size,
        has_grad_branch=grad_branch is not None,
        has_grad_residual0=grad_residual0 is not None,
        has_grad_residual1=grad_residual1 is not None,
        has_grad_beta=grad_beta is not None,
        has_grad_keep=grad_keep_out is not None,
        bf16_parity=bool(bf16_parity),
        block_h=row_block_h,
        num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_ROW_NUM_WARPS", 4),
    )

    grad_history_attended = torch.empty_like(history_attended)
    grad_current = torch.empty_like(current_streams)
    partial_sums = torch.empty(
        (rows, num_post_blocks, 4),
        device=current_streams.device,
        dtype=torch.float32,
    )
    _fused_prefix_residual_full_post_backward_kernel[(rows, num_post_blocks)](
        history_attended=history_attended,
        current_streams=current_streams,
        grad_branch=grad_branch if grad_branch is not None else dummy,
        grad_residual0=grad_residual0 if grad_residual0 is not None else dummy,
        grad_residual1=grad_residual1 if grad_residual1 is not None else dummy,
        fused_weight=fused_weight,
        param_buf=param_buf,
        row_grads=row_grads,
        grad_history_attended_out=grad_history_attended,
        grad_current_out=grad_current,
        partial_sums=partial_sums,
        hidden_size=hidden_size,
        num_blocks=num_post_blocks,
        has_grad_branch=grad_branch is not None,
        has_grad_residual0=grad_residual0 is not None,
        has_grad_residual1=grad_residual1 is not None,
        bf16_parity=bool(bf16_parity),
        block_h=post_block_h,
        num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_POST_NUM_WARPS", 4),
    )

    use_fused_finalize = _triton_backward_full_finalize_enabled()
    if use_fused_finalize:
        grad_history_logit = torch.empty_like(history_logit)
        grad_current_logits = torch.empty((rows, 2), device=current_streams.device, dtype=torch.float32)
        finalize_block_n = _pow2_block(
            num_post_blocks,
            default=1,
            minimum=1,
            maximum=max(1, num_post_blocks),
        )
        _fused_prefix_residual_full_finalize_backward_kernel[(rows,)](
            partial_sums=partial_sums,
            param_buf=param_buf,
            grad_history_logit_out=grad_history_logit,
            grad_current_logits_out=grad_current_logits,
            seq_len=seq_len,
            batch_size=batch_size,
            num_blocks=num_post_blocks,
            block_n=finalize_block_n,
            num_warps=1,
        )
        del partial_sums
    else:
        weight_sums = partial_sums.sum(dim=1)
        history_weight = param_buf[:, 5:7]
        current_weight = param_buf[:, 7:9]
        grad_history_weight = weight_sums[:, 0:2]
        grad_current_weight = weight_sums[:, 2:4]
        grad_hist_minus_current = grad_history_weight * history_weight * (1.0 - history_weight)
        grad_current_minus_hist = grad_current_weight * current_weight * (1.0 - current_weight)
        grad_history_logits = grad_hist_minus_current - grad_current_minus_hist
        grad_current_logits = (-grad_hist_minus_current + grad_current_minus_hist).contiguous()
        grad_history_logit = grad_history_logits.reshape(seq_len, batch_size, 2).permute(2, 1, 0).contiguous()
        del partial_sums, weight_sums

    grad_fused = torch.empty_like(fused_weight)
    weight_block_r = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_BLOCK_R", 1024),
        default=1024,
        maximum=max(16, rows),
    )
    weight_block_f = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_BLOCK_F", 8),
        default=8,
        minimum=8,
        maximum=max(16, flat_size),
    )
    _fused_prefix_residual_full_weight_backward_kernel[(triton.cdiv(flat_size, weight_block_f),)](
        history_attended=history_attended,
        current_streams=current_streams,
        param_buf=param_buf,
        row_grads=row_grads,
        grad_fused_out=grad_fused,
        total_rows=rows,
        hidden_size=hidden_size,
        bf16_parity=bool(bf16_parity),
        block_r=weight_block_r,
        block_f=weight_block_f,
        num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_WEIGHT_NUM_WARPS", 4),
    )

    if use_fused_finalize:
        scalar_grads = torch.empty((8,), device=static_alpha.device, dtype=static_alpha.dtype)
        scalar_block_r = _pow2_block(
            rows,
            default=1024,
            minimum=16,
            maximum=max(16, rows),
        )
        _fused_prefix_residual_full_scalar_backward_kernel[(8,)](
            row_grads=row_grads,
            scalar_grads_out=scalar_grads,
            total_rows=rows,
            block_r=scalar_block_r,
            num_warps=4,
        )
        grad_static_alpha = scalar_grads[0:2].view_as(static_alpha)
        grad_static_beta = scalar_grads[2:4].view_as(static_beta)
        grad_static_residual = scalar_grads[4:5].view_as(static_residual_param)
        grad_pre_branch_scale = scalar_grads[5:6].view_as(pre_branch_scale)
        grad_h_post_scale = scalar_grads[6:7].view_as(h_post_scale)
        grad_residual_scale = scalar_grads[7:8].view_as(residual_scale)
    else:
        scalar_grads = row_grads[:, 7:15].sum(dim=0)
        grad_static_alpha = scalar_grads[0:2].view_as(static_alpha).to(dtype=static_alpha.dtype)
        grad_static_beta = scalar_grads[2:4].view_as(static_beta).to(dtype=static_beta.dtype)
        grad_static_residual = scalar_grads[4].view_as(static_residual_param).to(dtype=static_residual_param.dtype)
        grad_pre_branch_scale = scalar_grads[5].view_as(pre_branch_scale).to(dtype=pre_branch_scale.dtype)
        grad_h_post_scale = scalar_grads[6].view_as(h_post_scale).to(dtype=h_post_scale.dtype)
        grad_residual_scale = scalar_grads[7].view_as(residual_scale).to(dtype=residual_scale.dtype)
    del row_grads

    key_block_r = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_BLOCK_R", 64),
        default=64,
        maximum=max(16, rows),
    )
    key_block_h = _pow2_block(
        _env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_BLOCK_H", 64),
        default=64,
        maximum=max(16, hidden_size),
    )
    grad_current_key = torch.empty_like(current_key_streams)
    grad_query_float = torch.zeros(query.shape, device=query.device, dtype=torch.float32)
    if recompute_current_key:
        current_inv_rms = torch.empty((rows * 2,), device=current_streams.device, dtype=torch.float32)
        inv_block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
            default=128,
            maximum=max(16, hidden_size),
        )
        _fixed_rms_inv_kernel[(rows * 2,)](
            x=current_streams,
            inv_rms_out=current_inv_rms,
            hidden_size=hidden_size,
            eps=float(norm_eps),
            block_h=inv_block_h,
            num_warps=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
        )
    else:
        current_inv_rms = current_streams
    _fused_prefix_residual_full_key_query_backward_kernel[
        (triton.cdiv(rows, key_block_r), triton.cdiv(hidden_size, key_block_h))
    ](
        current_key_streams=current_key_streams,
        current_streams=current_streams,
        current_inv_rms=current_inv_rms,
        query=query,
        grad_current_logits=grad_current_logits,
        grad_current_key_out=grad_current_key,
        grad_query_out=grad_query_float,
        hidden_size=hidden_size,
        total_rows=rows,
        cross_key=bool(cross_key),
        recompute_key=bool(recompute_current_key),
        block_r=key_block_r,
        block_h=key_block_h,
        num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_FULL_KEY_NUM_WARPS", 4),
    )
    grad_query = grad_query_float.to(dtype=query.dtype)
    return (
        grad_history_logit,
        grad_history_attended,
        grad_current,
        grad_current_key,
        grad_query,
        grad_fused,
        grad_static_alpha,
        grad_static_beta,
        grad_static_residual,
        grad_pre_branch_scale,
        grad_h_post_scale,
        grad_residual_scale,
    )


def _backward_torch(
    grad_branch: Optional[torch.Tensor],
    grad_residual0: Optional[torch.Tensor],
    grad_residual1: Optional[torch.Tensor],
    grad_beta: Optional[torch.Tensor],
    grad_keep_out: Optional[torch.Tensor],
    history_logit: torch.Tensor,
    history_attended: torch.Tensor,
    current_streams: torch.Tensor,
    current_key_streams: torch.Tensor,
    query: torch.Tensor,
    fused_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    static_residual_param: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    cross_key: bool,
) -> tuple[torch.Tensor, ...]:
    seq_len, batch_size, _, hidden_size = current_streams.shape
    flat_size = 2 * hidden_size
    compute_dtype = torch.float32
    hist_logit_sbd = history_logit.permute(2, 1, 0).to(dtype=compute_dtype)
    history_attended_f = history_attended.to(dtype=compute_dtype)
    current_f = current_streams.to(dtype=compute_dtype)
    current_key_f = current_key_streams.to(dtype=compute_dtype)
    query_f = query.to(dtype=compute_dtype)
    fused_f = fused_weight.to(dtype=compute_dtype)
    static_alpha_f = static_alpha.reshape(2).to(dtype=compute_dtype)
    static_beta_f = static_beta.reshape(2).to(dtype=compute_dtype)
    static_residual_f = static_residual_param.reshape(()).to(dtype=compute_dtype)
    pre_scale = pre_branch_scale.reshape(()).to(dtype=compute_dtype)
    post_scale = h_post_scale.reshape(()).to(dtype=compute_dtype)
    res_scale = residual_scale.reshape(()).to(dtype=compute_dtype)

    if cross_key:
        key0 = current_key_f[:, :, 1, :]
        key1 = current_key_f[:, :, 0, :]
    else:
        key0 = current_key_f[:, :, 0, :]
        key1 = current_key_f[:, :, 1, :]
    current_logit0 = (key0 * query_f[0].view(1, 1, hidden_size)).sum(dim=-1)
    current_logit1 = (key1 * query_f[1].view(1, 1, hidden_size)).sum(dim=-1)
    current_logits = torch.stack((current_logit0, current_logit1), dim=-1)
    history_weight = (hist_logit_sbd - current_logits).sigmoid()
    current_weight = 1.0 - history_weight
    attended = history_attended_f * history_weight.unsqueeze(-1) + current_f * current_weight.unsqueeze(-1)

    flat_attended = attended.reshape(seq_len, batch_size, flat_size)
    flat_current = current_f.reshape(seq_len, batch_size, flat_size)
    post_norm_eps = float(norm_eps) / max(1, flat_size)
    inv_attended = torch.rsqrt(flat_attended.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
    inv_current = torch.rsqrt(flat_current.pow(2).mean(dim=-1, keepdim=True) + post_norm_eps)
    dot_ab = torch.matmul(flat_attended, fused_f[:, :4])
    z_ab = dot_ab * inv_attended
    alpha = (z_ab[..., :2] * pre_scale + static_alpha_f.view(1, 1, 2)).sigmoid()
    beta = (z_ab[..., 2:4] * post_scale + static_beta_f.view(1, 1, 2)).sigmoid() * 2.0
    dot_keep = torch.matmul(flat_current, fused_f[:, 4:5])
    z_keep = dot_keep * inv_current
    keep = (z_keep.squeeze(-1) * res_scale + static_residual_f).sigmoid()

    grad_branch_f = _as_zeros_like(grad_branch, attended[:, :, 0, :]).to(dtype=compute_dtype)
    grad_residual0_f = _as_zeros_like(grad_residual0, current_f[:, :, 0, :]).to(dtype=compute_dtype)
    grad_residual1_f = _as_zeros_like(grad_residual1, current_f[:, :, 1, :]).to(dtype=compute_dtype)
    grad_beta_f = _as_zeros_like(grad_beta, beta).to(dtype=compute_dtype)
    if grad_keep_out is None:
        grad_keep_f = torch.zeros_like(keep)
    else:
        grad_keep_f = grad_keep_out.to(dtype=compute_dtype)

    grad_alpha = torch.stack(
        (
            (grad_branch_f * attended[:, :, 0, :]).sum(dim=-1),
            (grad_branch_f * attended[:, :, 1, :]).sum(dim=-1),
        ),
        dim=-1,
    )
    grad_attended_branch = torch.stack(
        (
            grad_branch_f * alpha[..., 0:1],
            grad_branch_f * alpha[..., 1:2],
        ),
        dim=2,
    )

    grad_y_alpha = grad_alpha * alpha * (1.0 - alpha)
    beta_sigmoid = beta * 0.5
    grad_y_beta = grad_beta_f * (2.0 * beta_sigmoid * (1.0 - beta_sigmoid))
    grad_pre_branch_scale = (grad_y_alpha * z_ab[..., :2]).sum().view_as(pre_branch_scale)
    grad_h_post_scale = (grad_y_beta * z_ab[..., 2:4]).sum().view_as(h_post_scale)
    grad_static_alpha = grad_y_alpha.sum(dim=(0, 1)).view_as(static_alpha)
    grad_static_beta = grad_y_beta.sum(dim=(0, 1)).view_as(static_beta)
    grad_z_ab = torch.cat((grad_y_alpha * pre_scale, grad_y_beta * post_scale), dim=-1)

    grad_dot_ab = grad_z_ab * inv_attended
    grad_inv_attended = (grad_z_ab * dot_ab).sum(dim=-1, keepdim=True)
    rows = seq_len * batch_size
    flat_attended_2d = flat_attended.reshape(rows, flat_size)
    grad_dot_ab_2d = grad_dot_ab.reshape(rows, 4)
    grad_fused_ab = flat_attended_2d.t().matmul(grad_dot_ab_2d)

    current0 = current_f[:, :, 0, :]
    current1 = current_f[:, :, 1, :]
    keep_expanded = keep.unsqueeze(-1)
    one_minus_keep = 1.0 - keep_expanded
    grad_current_res0 = grad_residual0_f * keep_expanded + grad_residual1_f * one_minus_keep
    grad_current_res1 = grad_residual0_f * one_minus_keep + grad_residual1_f * keep_expanded
    grad_keep_f = grad_keep_f + (
        grad_residual0_f * (current0 - current1) + grad_residual1_f * (current1 - current0)
    ).sum(dim=-1)
    grad_y_keep = grad_keep_f * keep * (1.0 - keep)
    grad_residual_scale = (grad_y_keep * z_keep.squeeze(-1)).sum().view_as(residual_scale)
    grad_static_residual = grad_y_keep.sum().view_as(static_residual_param)
    grad_z_keep = grad_y_keep.unsqueeze(-1) * res_scale
    grad_dot_keep = grad_z_keep * inv_current
    grad_inv_current = grad_z_keep * dot_keep
    flat_current_2d = flat_current.reshape(rows, flat_size)
    grad_fused_keep = flat_current_2d.t().matmul(grad_dot_keep.reshape(rows, 1))
    if _triton_backward_post_enabled():
        post_block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_BACKWARD_POST_BLOCK_H", 256),
            default=256,
            maximum=max(16, hidden_size),
        )
        num_blocks = triton.cdiv(hidden_size, post_block_h)
        partial_sums = torch.empty((rows, num_blocks, 4), device=current_streams.device, dtype=torch.float32)
        grad_history_attended = torch.empty_like(history_attended)
        grad_current = torch.empty_like(current_streams)
        _fused_prefix_residual_post_backward_kernel[(rows, num_blocks)](
            history_attended=history_attended,
            current_streams=current_streams,
            grad_branch=grad_branch_f,
            grad_residual0=grad_residual0_f,
            grad_residual1=grad_residual1_f,
            fused_weight=fused_weight,
            grad_dot_ab=grad_dot_ab.reshape(rows, 4).contiguous(),
            grad_inv_attended=grad_inv_attended.reshape(rows).contiguous(),
            inv_attended=inv_attended.reshape(rows).contiguous(),
            grad_dot_keep=grad_dot_keep.reshape(rows).contiguous(),
            grad_inv_current=grad_inv_current.reshape(rows).contiguous(),
            inv_current=inv_current.reshape(rows).contiguous(),
            history_weight=history_weight.contiguous(),
            current_weight=current_weight.contiguous(),
            alpha=alpha.contiguous(),
            keep=keep.contiguous(),
            grad_history_attended_out=grad_history_attended,
            grad_current_out=grad_current,
            partial_sums=partial_sums,
            hidden_size=hidden_size,
            flat_size=flat_size,
            num_blocks=num_blocks,
            block_h=post_block_h,
            num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_POST_NUM_WARPS", 4),
        )
        weight_sums = partial_sums.sum(dim=1).reshape(seq_len, batch_size, 4)
        grad_history_weight = weight_sums[..., 0:2]
        grad_current_weight = weight_sums[..., 2:4]
        del partial_sums
    else:
        grad_flat_attended = grad_dot_ab.matmul(fused_f[:, :4].t())
        grad_flat_attended = grad_flat_attended - (
            grad_inv_attended * inv_attended.pow(3) * flat_attended / float(flat_size)
        )
        grad_flat_current = grad_dot_keep.matmul(fused_f[:, 4:5].t())
        grad_flat_current = grad_flat_current - (
            grad_inv_current * inv_current.pow(3) * flat_current / float(flat_size)
        )
        grad_attended = grad_attended_branch + grad_flat_attended.reshape_as(attended)
        grad_current = torch.stack((grad_current_res0, grad_current_res1), dim=2) + grad_flat_current.reshape_as(current_f)
        grad_history_attended = grad_attended * history_weight.unsqueeze(-1)
        grad_current = grad_current + grad_attended * current_weight.unsqueeze(-1)
        grad_history_weight = (grad_attended * history_attended_f).sum(dim=-1)
        grad_current_weight = (grad_attended * current_f).sum(dim=-1)
    grad_hist_minus_current = grad_history_weight * history_weight * (1.0 - history_weight)
    grad_current_minus_hist = grad_current_weight * current_weight * (1.0 - current_weight)
    grad_hist_logit_sbd = grad_hist_minus_current - grad_current_minus_hist
    grad_current_logits = -grad_hist_minus_current + grad_current_minus_hist

    if _triton_backward_key_query_enabled():
        key_query_block_r = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_BACKWARD_KEY_QUERY_BLOCK_R", 32),
            default=32,
            maximum=max(16, rows),
        )
        key_query_block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_BACKWARD_KEY_QUERY_BLOCK_H", 128),
            default=128,
            maximum=max(16, hidden_size),
        )
        key_query_num_row_blocks = triton.cdiv(rows, key_query_block_r)
        key_query_num_h_blocks = triton.cdiv(hidden_size, key_query_block_h)
        grad_query_partials = torch.empty(
            (key_query_num_row_blocks, 2, key_query_num_h_blocks * key_query_block_h),
            device=current_streams.device,
            dtype=torch.float32,
        )
        grad_current_key = torch.empty_like(current_key_f)
        _fused_prefix_residual_key_query_backward_kernel[(key_query_num_row_blocks, key_query_num_h_blocks)](
            current_key_streams=current_key_streams,
            query=query,
            grad_current_logits=grad_current_logits.contiguous(),
            grad_current_key_out=grad_current_key,
            grad_query_partials=grad_query_partials,
            hidden_size=hidden_size,
            total_rows=rows,
            num_h_blocks=key_query_num_h_blocks,
            cross_key=bool(cross_key),
            block_r=key_query_block_r,
            block_h=key_query_block_h,
            num_warps=_env_int("MEGATRON_DUAL_TRITON_BACKWARD_KEY_QUERY_NUM_WARPS", 4),
        )
        grad_query = grad_query_partials.sum(dim=0)[:, :hidden_size].contiguous()
        del grad_query_partials
    else:
        grad_query = torch.zeros_like(query_f)
        grad_current_key = torch.zeros_like(current_key_f)
        grad_cl0 = grad_current_logits[:, :, 0].unsqueeze(-1)
        grad_cl1 = grad_current_logits[:, :, 1].unsqueeze(-1)
        if cross_key:
            grad_current_key[:, :, 1, :] = grad_current_key[:, :, 1, :] + grad_cl0 * query_f[0].view(1, 1, hidden_size)
            grad_current_key[:, :, 0, :] = grad_current_key[:, :, 0, :] + grad_cl1 * query_f[1].view(1, 1, hidden_size)
            grad_query[0] = (grad_cl0 * current_key_f[:, :, 1, :]).sum(dim=(0, 1))
            grad_query[1] = (grad_cl1 * current_key_f[:, :, 0, :]).sum(dim=(0, 1))
        else:
            grad_current_key[:, :, 0, :] = grad_current_key[:, :, 0, :] + grad_cl0 * query_f[0].view(1, 1, hidden_size)
            grad_current_key[:, :, 1, :] = grad_current_key[:, :, 1, :] + grad_cl1 * query_f[1].view(1, 1, hidden_size)
            grad_query[0] = (grad_cl0 * current_key_f[:, :, 0, :]).sum(dim=(0, 1))
            grad_query[1] = (grad_cl1 * current_key_f[:, :, 1, :]).sum(dim=(0, 1))

    grad_fused = torch.cat((grad_fused_ab, grad_fused_keep), dim=1)
    grad_history_logit = grad_hist_logit_sbd.permute(2, 1, 0).contiguous()
    return (
        grad_history_logit.to(dtype=history_logit.dtype),
        grad_history_attended.to(dtype=history_attended.dtype),
        grad_current.to(dtype=current_streams.dtype),
        grad_current_key.to(dtype=current_key_streams.dtype),
        grad_query.to(dtype=query.dtype),
        grad_fused.to(dtype=fused_weight.dtype),
        grad_static_alpha.to(dtype=static_alpha.dtype),
        grad_static_beta.to(dtype=static_beta.dtype),
        grad_static_residual.to(dtype=static_residual_param.dtype),
        grad_pre_branch_scale.to(dtype=pre_branch_scale.dtype),
        grad_h_post_scale.to(dtype=h_post_scale.dtype),
        grad_residual_scale.to(dtype=residual_scale.dtype),
    )


class _TritonFixedRMSNorm(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float, block_h_override: int, num_warps_override: int) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        x = x.contiguous()
        hidden_size = int(x.shape[-1])
        rows = int(x.numel() // hidden_size)
        output = torch.empty_like(x)
        inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        requested_block_h = block_h_override or _env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128)
        block_h = _pow2_block(
            requested_block_h,
            default=128,
            maximum=max(16, hidden_size),
        )
        num_warps = num_warps_override or _env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4)
        _fixed_rms_norm_forward_kernel[(rows,)](
            x=x,
            output=output,
            inv_rms_out=inv_rms,
            hidden_size=hidden_size,
            eps=float(eps),
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.hidden_size = hidden_size
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.save_for_backward(x, inv_rms)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, inv_rms = ctx.saved_tensors
        hidden_size = ctx.hidden_size
        rows = int(x.numel() // hidden_size)
        grad_x = torch.empty_like(x)
        _fixed_rms_norm_backward_kernel[(rows,)](
            grad_output=grad_output.contiguous(),
            x=x,
            inv_rms=inv_rms,
            grad_x=grad_x,
            hidden_size=hidden_size,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        return grad_x, None, None, None


def triton_fixed_rms_norm(
    x: torch.Tensor,
    eps: float,
    *,
    block_h: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise ValueError("triton_fixed_rms_norm requires a CUDA tensor")
    return _TritonFixedRMSNorm.apply(x, float(eps), int(block_h or 0), int(num_warps or 0))


def triton_dual_history_key_pack_norm(
    packed_blocks: torch.Tensor,
    eps: float,
    *,
    cross_key: bool,
    block_h: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE or not packed_blocks.is_cuda:
        raise RuntimeError("Triton Dual history-key pack norm requires CUDA")
    if packed_blocks.ndim != 5 or int(packed_blocks.shape[3]) != 2:
        raise ValueError("expected packed blocks [history, seq, batch, 2, hidden]")
    packed_blocks = packed_blocks.contiguous()
    history_size, seq_len, batch_size, _, hidden_size = [int(x) for x in packed_blocks.shape]
    keys = torch.empty(
        (2, history_size, batch_size, seq_len, hidden_size),
        device=packed_blocks.device,
        dtype=packed_blocks.dtype,
    )
    block_h = _pow2_block(block_h, default=128, maximum=max(16, hidden_size))
    rows = history_size * seq_len * batch_size * 2
    _dual_history_key_pack_norm_kernel[(rows,)](
        packed_blocks=packed_blocks,
        keys_out=keys,
        history_size=history_size,
        seq_len=seq_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        eps=float(eps),
        cross_key=bool(cross_key),
        block_h=block_h,
        num_warps=num_warps,
    )
    return keys


class _TritonDualHistoryAppendPack(Function):
    @staticmethod
    def forward(
        ctx,
        prev_keys: torch.Tensor,
        prev_values: torch.Tensor,
        block: torch.Tensor,
        eps: float,
        cross_key: bool,
        cross_value: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_size = int(prev_keys.shape[1])
        seq_len, batch_size, _, hidden_size = [int(value) for value in block.shape]
        output_history_size = history_size + 1
        output_shape = (2, output_history_size, batch_size, seq_len, hidden_size)
        keys = torch.empty(output_shape, device=block.device, dtype=block.dtype)
        values = torch.empty_like(keys)
        rows = seq_len * batch_size * 2
        inv_rms = torch.empty((rows,), device=block.device, dtype=torch.float32)

        if history_size:
            prefix_elements = int(prev_keys.numel())
            rows_per_history = batch_size * seq_len * hidden_size
            copy_block = _pow2_block(
                _env_int("MEGATRON_DUAL_TRITON_HISTORY_APPEND_COPY_BLOCK", 256),
                default=256,
                maximum=4096,
            )
            _dual_history_append_prefix_copy_kernel[(triton.cdiv(prefix_elements, copy_block),)](
                prev_keys=prev_keys,
                prev_values=prev_values,
                keys_out=keys,
                values_out=values,
                prefix_elements=prefix_elements,
                output_history_size=output_history_size,
                history_size=history_size,
                rows_per_history=rows_per_history,
                block=copy_block,
                num_warps=4,
            )

        block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_HISTORY_APPEND_BLOCK_H", 128),
            default=128,
            maximum=max(16, hidden_size),
        )
        num_warps = _env_int("MEGATRON_DUAL_TRITON_HISTORY_APPEND_NUM_WARPS", 4)
        _dual_history_append_new_forward_kernel[(rows,)](
            block=block,
            keys_out=keys,
            values_out=values,
            inv_rms_out=inv_rms,
            output_history_size=output_history_size,
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            eps=float(eps),
            cross_key=bool(cross_key),
            cross_value=bool(cross_value),
            block_h=block_h,
            num_warps=num_warps,
        )
        ctx.history_size = history_size
        ctx.output_history_size = output_history_size
        ctx.seq_len = seq_len
        ctx.batch_size = batch_size
        ctx.hidden_size = hidden_size
        ctx.cross_key = bool(cross_key)
        ctx.cross_value = bool(cross_value)
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.save_for_backward(block, inv_rms)
        return keys, values

    @staticmethod
    def backward(ctx, grad_keys: torch.Tensor, grad_values: torch.Tensor):
        block, inv_rms = ctx.saved_tensors
        grad_keys = grad_keys.contiguous()
        grad_values = grad_values.contiguous()
        grad_block = torch.empty_like(block)
        rows = ctx.seq_len * ctx.batch_size * 2
        _dual_history_append_new_backward_kernel[(rows,)](
            grad_keys=grad_keys,
            grad_values=grad_values,
            block=block,
            inv_rms=inv_rms,
            grad_block=grad_block,
            output_history_size=ctx.output_history_size,
            seq_len=ctx.seq_len,
            batch_size=ctx.batch_size,
            hidden_size=ctx.hidden_size,
            cross_key=ctx.cross_key,
            cross_value=ctx.cross_value,
            block_h=ctx.block_h,
            num_warps=ctx.num_warps,
        )
        grad_prev_keys = grad_keys[:, : ctx.history_size]
        grad_prev_values = grad_values[:, : ctx.history_size]
        return grad_prev_keys, grad_prev_values, grad_block, None, None, None


def triton_dual_history_append_pack(
    prev_keys: torch.Tensor,
    prev_values: torch.Tensor,
    block: torch.Tensor,
    eps: float,
    cross_key: bool,
    cross_value: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (prev_keys.is_cuda and prev_values.is_cuda and block.is_cuda):
        raise ValueError("triton_dual_history_append_pack requires CUDA tensors")
    if prev_keys.ndim != 5 or prev_values.shape != prev_keys.shape:
        raise ValueError("expected matching previous key/value tensors [2, history, batch, seq, hidden]")
    if block.ndim != 4 or int(block.shape[2]) != 2:
        raise ValueError("expected block [seq, batch, 2, hidden]")
    expected_tail = (int(block.shape[1]), int(block.shape[0]), int(block.shape[3]))
    if int(prev_keys.shape[0]) != 2 or tuple(int(value) for value in prev_keys.shape[2:]) != expected_tail:
        raise ValueError("previous key/value layout does not match the block")
    if not (prev_keys.dtype == prev_values.dtype == block.dtype == torch.bfloat16):
        raise ValueError("triton_dual_history_append_pack currently requires BF16 tensors")
    if not (prev_keys.is_contiguous() and prev_values.is_contiguous() and block.is_contiguous()):
        raise ValueError("triton_dual_history_append_pack requires contiguous tensors")
    return _TritonDualHistoryAppendPack.apply(
        prev_keys,
        prev_values,
        block,
        float(eps),
        bool(cross_key),
        bool(cross_value),
    )


def _triton_fixed_rms_norm_recompute(
    x: torch.Tensor,
    eps: float,
    *,
    block_h: int,
    num_warps: int,
) -> torch.Tensor:
    x = x.contiguous()
    hidden_size = int(x.shape[-1])
    rows = int(x.numel() // hidden_size)
    output = torch.empty_like(x)
    block_h = _pow2_block(block_h, default=128, maximum=max(16, hidden_size))
    _fixed_rms_norm_recompute_kernel[(rows,)](
        x=x,
        output=output,
        hidden_size=hidden_size,
        eps=float(eps),
        block_h=block_h,
        num_warps=num_warps,
    )
    return output


class _FusedDualPrefixResidualRead(Function):
    @staticmethod
    def forward(
        ctx,
        history_logit: torch.Tensor,
        history_attended: torch.Tensor,
        current_streams: torch.Tensor,
        current_key_streams: torch.Tensor,
        query: torch.Tensor,
        fused_weight: torch.Tensor,
        static_alpha: torch.Tensor,
        static_beta: torch.Tensor,
        static_residual_param: torch.Tensor,
        pre_branch_scale: torch.Tensor,
        h_post_scale: torch.Tensor,
        residual_scale: torch.Tensor,
        norm_eps: float,
        cross_key: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        history_logit = history_logit.contiguous()
        history_attended = history_attended.contiguous()
        current_streams = current_streams.contiguous()
        current_key_streams = current_key_streams.contiguous()
        query = query.contiguous()
        fused_weight = fused_weight.contiguous()
        static_alpha = static_alpha.contiguous()
        static_beta = static_beta.contiguous()
        static_residual_param = static_residual_param.contiguous()
        pre_branch_scale = pre_branch_scale.contiguous()
        h_post_scale = h_post_scale.contiguous()
        residual_scale = residual_scale.contiguous()

        seq_len, batch_size, num_streams, hidden_size = [int(x) for x in current_streams.shape]
        if num_streams != 2:
            raise ValueError(f"Expected 2 streams, got {num_streams}")
        flat_size = int(2 * hidden_size)
        total_rows = int(seq_len * batch_size)
        params = torch.empty((total_rows, 16), device=current_streams.device, dtype=torch.float32)
        branch = torch.empty((seq_len, batch_size, hidden_size), device=current_streams.device, dtype=current_streams.dtype)
        residual0 = torch.empty_like(branch)
        residual1 = torch.empty_like(branch)
        beta = torch.empty((seq_len, batch_size, 2), device=current_streams.device, dtype=current_streams.dtype)
        keep = torch.empty((seq_len, batch_size), device=current_streams.device, dtype=current_streams.dtype)
        params_block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_PARAMS_BLOCK_H", 256),
            default=256,
            maximum=max(16, hidden_size),
        )
        apply_block_h = _pow2_block(
            _env_int("MEGATRON_DUAL_TRITON_APPLY_BLOCK_H", 512),
            default=512,
            maximum=max(16, hidden_size),
        )
        params_num_warps = _env_int("MEGATRON_DUAL_TRITON_PARAMS_NUM_WARPS", 4)
        apply_num_warps = _env_int("MEGATRON_DUAL_TRITON_APPLY_NUM_WARPS", 4)
        bf16_parity = os.environ.get("MEGATRON_DUAL_TRITON_BF16_PARITY", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        params_grid = (total_rows,)
        apply_grid = (total_rows, triton.cdiv(hidden_size, apply_block_h))
        params_args = [
            ("history_logit", history_logit),
            ("history_attended", history_attended),
            ("current_streams", current_streams),
            ("current_key_streams", current_key_streams),
            ("query", query),
            ("fused_weight", fused_weight),
            ("static_alpha_flat", static_alpha.reshape(-1)),
            ("static_beta_flat", static_beta.reshape(-1)),
            ("static_residual_flat", static_residual_param.reshape(-1)),
            ("pre_branch_scale_flat", pre_branch_scale.reshape(-1)),
            ("h_post_scale_flat", h_post_scale.reshape(-1)),
            ("residual_scale_flat", residual_scale.reshape(-1)),
            ("params", params),
            ("seq_len", seq_len),
            ("batch_size", batch_size),
            ("hidden_size", hidden_size),
            ("flat_size", flat_size),
            ("eps", float(norm_eps) / max(1, flat_size)),
            ("cross_key", bool(cross_key)),
            ("bf16_parity", bf16_parity),
            ("block_h", params_block_h),
        ]
        _maybe_print_launch_debug("params", params_grid, params_args)
        _fused_prefix_residual_params_kernel[params_grid](
            history_logit=history_logit,
            history_attended=history_attended,
            current_streams=current_streams,
            current_key_streams=current_key_streams,
            query=query,
            fused_weight=fused_weight,
            static_alpha=static_alpha.reshape(-1),
            static_beta=static_beta.reshape(-1),
            static_residual_param=static_residual_param.reshape(-1),
            pre_branch_scale=pre_branch_scale.reshape(-1),
            h_post_scale=h_post_scale.reshape(-1),
            residual_scale=residual_scale.reshape(-1),
            param_buf=params,
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            flat_size=flat_size,
            eps=float(norm_eps) / max(1, flat_size),
            cross_key=bool(cross_key),
            bf16_parity=bf16_parity,
            block_h=params_block_h,
            num_warps=params_num_warps,
        )
        apply_args = [
            ("history_attended", history_attended),
            ("current_streams", current_streams),
            ("params", params),
            ("branch", branch),
            ("residual0", residual0),
            ("residual1", residual1),
            ("beta", beta),
            ("keep", keep),
            ("hidden_size", hidden_size),
            ("bf16_parity", bf16_parity),
            ("block_h", apply_block_h),
        ]
        _maybe_print_launch_debug("apply", apply_grid, apply_args)
        _fused_prefix_residual_apply_kernel[apply_grid](
            history_attended=history_attended,
            current_streams=current_streams,
            param_buf=params,
            branch_out=branch,
            residual0_out=residual0,
            residual1_out=residual1,
            beta_out=beta,
            keep_out=keep,
            hidden_size=hidden_size,
            bf16_parity=bf16_parity,
            block_h=apply_block_h,
            num_warps=apply_num_warps,
        )
        ctx.norm_eps = float(norm_eps)
        ctx.cross_key = bool(cross_key)
        ctx.bf16_parity = bf16_parity
        ctx.full_backward = _triton_backward_full_enabled()
        ctx.recompute_current_key_streams = os.environ.get(
            "MEGATRON_DUAL_RECOMPUTE_CURRENT_KEY_STREAMS", "0"
        ).lower() in {"1", "true", "yes", "on"}
        saved_tensors = (
            history_logit,
            history_attended,
            current_streams,
            query,
            fused_weight,
            static_alpha,
            static_beta,
            static_residual_param,
            pre_branch_scale,
            h_post_scale,
            residual_scale,
        )
        if not ctx.recompute_current_key_streams:
            saved_tensors = (*saved_tensors[:3], current_key_streams, *saved_tensors[3:])
        if ctx.full_backward:
            saved_tensors = (*saved_tensors, params)
        ctx.save_for_backward(*saved_tensors)
        return branch, residual0, residual1, beta, keep

    @staticmethod
    def backward(
        ctx,
        grad_branch: Optional[torch.Tensor],
        grad_residual0: Optional[torch.Tensor],
        grad_residual1: Optional[torch.Tensor],
        grad_beta: Optional[torch.Tensor],
        grad_keep: Optional[torch.Tensor],
    ):
        if ctx.full_backward:
            *saved, param_buf = ctx.saved_tensors
        else:
            saved = list(ctx.saved_tensors)
            param_buf = None
        if ctx.recompute_current_key_streams:
            current_streams = saved[2]
            if ctx.full_backward:
                current_key_streams = current_streams
            else:
                current_key_streams = _triton_fixed_rms_norm_recompute(
                    current_streams,
                    ctx.norm_eps,
                    block_h=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_BLOCK_H", 128),
                    num_warps=_env_int("MEGATRON_DUAL_TRITON_CURRENT_KEY_NORM_NUM_WARPS", 4),
                )
            saved = [*saved[:3], current_key_streams, *saved[3:]]
        if ctx.full_backward:
            grads = _backward_triton_full(
                grad_branch,
                grad_residual0,
                grad_residual1,
                grad_beta,
                grad_keep,
                *saved,
                param_buf,
                ctx.norm_eps,
                ctx.cross_key,
                ctx.bf16_parity,
                recompute_current_key=ctx.recompute_current_key_streams,
            )
        else:
            grads = _backward_torch(
                grad_branch,
                grad_residual0,
                grad_residual1,
                grad_beta,
                grad_keep,
                *saved,
                ctx.norm_eps,
                ctx.cross_key,
            )
        return (*grads, None, None)


def triton_fused_dual_prefix_residual_read(
    history_prefix: tuple[torch.Tensor, torch.Tensor],
    current_streams: torch.Tensor,
    current_key_streams: torch.Tensor,
    query: torch.Tensor,
    fused_weight: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor,
    static_residual_param: torch.Tensor,
    pre_branch_scale: torch.Tensor,
    h_post_scale: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_eps: float,
    *,
    cross_key: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    history_logit, history_attended = history_prefix
    if not current_streams.is_cuda:
        raise ValueError("triton_fused_dual_prefix_residual_read requires CUDA tensors")
    if current_streams.ndim != 4 or current_streams.shape[2] != 2:
        raise ValueError(f"Expected current_streams [S, B, 2, H], got {tuple(current_streams.shape)}")
    return _FusedDualPrefixResidualRead.apply(
        history_logit,
        history_attended,
        current_streams,
        current_key_streams,
        query,
        fused_weight,
        static_alpha,
        static_beta,
        static_residual_param,
        pre_branch_scale,
        h_post_scale,
        residual_scale,
        float(norm_eps),
        bool(cross_key),
    )
