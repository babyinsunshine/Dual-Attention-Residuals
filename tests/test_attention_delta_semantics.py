from __future__ import annotations

import torch
import torch.nn as nn

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
