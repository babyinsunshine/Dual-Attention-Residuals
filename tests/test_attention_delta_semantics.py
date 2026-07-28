from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from dual_attentions_residuals.dual_residual_triton import (
    triton_dual_te_attention_epilogue,
    triton_te_attention_delta,
)
from dual_attentions_residuals.model import MCoreAtomicLayer


class _RawSelfAttention(nn.Module):
    def __init__(self, raw_value: float) -> None:
        super().__init__()
        self.raw_value = raw_value

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return torch.full_like(hidden_states, self.raw_value), None


class _ResidualAddingAttentionLayer(nn.Module):
    def __init__(self, raw_value: float) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attention = _RawSelfAttention(raw_value)
        self.hidden_dropout = 0.0
        self.recompute_input_layernorm = False

    def _forward_attention(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        raw, _ = self.self_attention(hidden_states, attention_mask, **kwargs)
        return hidden_states + raw, None


def _make_atomic_layer(raw_value: float) -> MCoreAtomicLayer:
    adapter = MCoreAtomicLayer.__new__(MCoreAtomicLayer)
    nn.Module.__init__(adapter)
    adapter.sequence_batch_size = 1
    adapter.layer = _ResidualAddingAttentionLayer(raw_value)
    adapter.last_aux_loss = None
    return adapter


def test_attention_delta_returns_raw_branch_without_bf16_residual_round_trip() -> None:
    hidden_states = torch.full((2, 1, 8), 16.0, dtype=torch.bfloat16)
    expected_raw = torch.full_like(hidden_states, 0.03125)
    adapter = _make_atomic_layer(raw_value=0.03125)

    actual = adapter.attention_delta(hidden_states)

    torch.testing.assert_close(actual, expected_raw, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_attention_delta_preserves_small_raw_bf16_updates() -> None:
    attention_input = torch.full(
        (2, 2, 128), 16.0, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    raw_attention = torch.full(
        (2, 2, 128), 0.03125, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    actual = triton_te_attention_delta(attention_input, raw_attention)

    torch.testing.assert_close(actual, raw_attention, rtol=0.0, atol=0.0)
    actual.backward(torch.ones_like(actual))
    assert attention_input.grad is None
    torch.testing.assert_close(
        raw_attention.grad, torch.ones_like(raw_attention), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dual_triton_epilogue_uses_raw_delta_for_forward_and_beta_gradient(
    has_residual: bool,
) -> None:
    shape = (2, 2, 128)
    attention_input = torch.full(
        shape, 16.0, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    raw_attention = torch.full(
        shape, 0.03125, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    residual0 = torch.full(
        shape, 1.0, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    residual1 = torch.full(
        shape, -1.0, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    beta = torch.empty((*shape[:-1], 2), device="cuda", dtype=torch.bfloat16)
    beta[..., 0] = 0.25
    beta[..., 1] = 0.75
    beta.requires_grad_()

    actual = triton_dual_te_attention_epilogue(
        attention_input,
        raw_attention,
        residual0,
        residual1,
        beta,
        has_residual=has_residual,
    )
    expected0 = raw_attention.float() * beta[..., 0:1].float()
    expected1 = raw_attention.float() * beta[..., 1:2].float()
    if has_residual:
        expected0 = expected0 + residual0.float()
        expected1 = expected1 + residual1.float()
    expected = torch.stack((expected0, expected1), dim=-2).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.backward(torch.ones_like(actual))

    assert attention_input.grad is None
    torch.testing.assert_close(
        raw_attention.grad, torch.ones_like(raw_attention), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        beta.grad, torch.full_like(beta, 4.0), rtol=0.0, atol=0.0
    )
    if has_residual:
        torch.testing.assert_close(
            residual0.grad, torch.ones_like(residual0), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            residual1.grad, torch.ones_like(residual1), rtol=0.0, atol=0.0
        )
    else:
        assert residual0.grad is None
        assert residual1.grad is None
