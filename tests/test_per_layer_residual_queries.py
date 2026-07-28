from __future__ import annotations

import unittest
from collections import Counter
from types import MethodType
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from dual_attentions_residuals import residual_blocks as rb


def make_config(variant: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            variant=variant,
            hidden_size=8,
            layernorm_epsilon=1.0e-5,
            hyper_conn_block_layers=3,
            hyper_conn_attres_separate_k_proj=False,
            num_layers=6,
            dual_kv_mode="cross",
        ),
        train=SimpleNamespace(bf16=False, fp16=False),
    )


def make_layers(count: int = 3) -> list[nn.Module]:
    return [FakeAtomicLayer() for _ in range(count)]


class FakeAtomicLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_aux_loss = None

    def attention_delta(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
    ) -> torch.Tensor:
        del attention_mask, rotary_pos_emb
        return hidden_states * 0.125

    def mlp_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 0.125


class IdentityNorm(nn.Module):
    eps = 1.0e-5
    force_fp32 = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class PerLayerResidualQueryStructureTests(unittest.TestCase):
    def test_attnres_owns_distinct_attention_and_mlp_query_per_layer(self) -> None:
        block = rb.AttentionResidualBlock(
            make_config("attention_residual_block"), make_layers()
        )
        block.attn_res_norm = IdentityNorm()
        block.mlp_res_norm = IdentityNorm()

        queries = [*block.attn_qs, *block.mlp_qs]
        self.assertEqual(len(block.attn_qs), 3)
        self.assertEqual(len(block.mlp_qs), 3)
        self.assertEqual([tuple(query.shape) for query in queries], [(8,)] * 6)
        self.assertEqual(len({query.data_ptr() for query in queries}), 6)
        self.assertTrue(all(torch.count_nonzero(query).item() == 0 for query in queries))
        self.assertTrue(all(query.sequence_parallel for query in queries))

        state_keys = set(block.state_dict())
        self.assertTrue({f"attn_qs.{index}" for index in range(3)} <= state_keys)
        self.assertTrue({f"mlp_qs.{index}" for index in range(3)} <= state_keys)
        self.assertNotIn("attn_res_proj.weight", state_keys)
        self.assertNotIn("mlp_res_proj.weight", state_keys)

    def test_dual_owns_distinct_attention_and_mlp_query_pair_per_layer(self) -> None:
        calls: list[int] = []

        def record_init(query: torch.Tensor, *, preferred_stream: int, balanced: bool) -> None:
            del balanced
            calls.append(preferred_stream)
            query.data.zero_()

        with mock.patch.object(rb, "init_dual_stream_query_", side_effect=record_init):
            block = rb.DualAttentionResidualBlock(
                make_config("dual_attention_residual_block"),
                make_layers(),
                layer_index=5,
            )

        query_pairs = [*block.attn_qs, *block.mlp_qs]
        query_rows = [query[row] for query in query_pairs for row in range(2)]
        self.assertEqual(len(block.attn_qs), 3)
        self.assertEqual(len(block.mlp_qs), 3)
        self.assertEqual([tuple(query.shape) for query in query_pairs], [(2, 8)] * 6)
        self.assertEqual(len({row.data_ptr() for row in query_rows}), 12)
        self.assertTrue(all(query.sequence_parallel for query in query_pairs))
        self.assertEqual(len(calls), 6)
        self.assertEqual(Counter(calls), Counter({1: 4, 0: 2}))

        state_keys = set(block.state_dict())
        self.assertTrue({f"attn_qs.{index}" for index in range(3)} <= state_keys)
        self.assertTrue({f"mlp_qs.{index}" for index in range(3)} <= state_keys)
        self.assertNotIn("attn_q", state_keys)
        self.assertNotIn("mlp_q", state_keys)


class PerLayerResidualQueryForwardTests(unittest.TestCase):
    @staticmethod
    def assert_all_nonzero_finite_gradients(queries: list[nn.Parameter]) -> None:
        for query in queries:
            if query.grad is None:
                raise AssertionError("query did not receive a gradient")
            if not torch.isfinite(query.grad).all():
                raise AssertionError("query gradient is not finite")
            if torch.count_nonzero(query.grad).item() == 0:
                raise AssertionError("query gradient is zero")

    def test_attnres_forward_uses_matching_query_and_all_queries_receive_gradients(self) -> None:
        block = rb.AttentionResidualBlock(
            make_config("attention_residual_block"), make_layers()
        )
        block.attn_res_norm = IdentityNorm()
        block.mlp_res_norm = IdentityNorm()
        observed: list[torch.Tensor] = []

        def record_read(self, cache, partial_block, query, **kwargs):
            del self, cache, kwargs
            observed.append(query)
            return partial_block + query.sum()

        block._block_attn_res = MethodType(record_read, block)
        x = torch.randn(2, 1, 8, requires_grad=True)
        with mock.patch.multiple(
            rb,
            _ATTRES_TRITON_HISTORY_APPEND_PACK=False,
            _ATTRES_CACHE_HISTORY_PREFIX_PAIR=False,
            _ATTRES_CACHE_HISTORY_LOGITS_PAIR=False,
            _ATTRES_TRITON_CURRENT_KEY_NORM=False,
            _RESIDUAL_TE_ATTENTION_EPILOGUE=False,
        ):
            output = block(x, state={})
        output.sum().backward()

        expected = [
            block.attn_qs[0],
            block.mlp_qs[0],
            block.attn_qs[1],
            block.mlp_qs[1],
            block.attn_qs[2],
            block.mlp_qs[2],
        ]
        self.assertEqual(len(observed), len(expected))
        self.assertTrue(all(actual is wanted for actual, wanted in zip(observed, expected)))
        self.assert_all_nonzero_finite_gradients(expected)

    def test_attnres_paired_prefix_is_computed_once_per_atomic_layer(self) -> None:
        block = rb.AttentionResidualBlock(
            make_config("attention_residual_block"), make_layers()
        )
        block.attn_res_norm = IdentityNorm()
        block.mlp_res_norm = IdentityNorm()
        observed_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_prefix(keys, values, attn_query, mlp_query, **kwargs):
            del keys, values, kwargs
            observed_pairs.append((attn_query, mlp_query))
            return None, None

        def passthrough_read(self, cache, partial_block, query, **kwargs):
            del self, cache, query, kwargs
            return partial_block

        block._block_attn_res = MethodType(passthrough_read, block)
        with mock.patch.multiple(
            rb,
            _ATTRES_TRITON_HISTORY_APPEND_PACK=False,
            _ATTRES_CACHE_HISTORY_PREFIX_PAIR=True,
            _ATTRES_RECOMPUTE_HISTORY_VALUES=False,
            _ATTRES_RECOMPUTE_HISTORY_KEYS=False,
            _ATTRES_TRITON_CURRENT_KEY_NORM=False,
            _RESIDUAL_TE_ATTENTION_EPILOGUE=False,
            _attres_history_prefix_pair=record_prefix,
        ):
            block(torch.randn(2, 1, 8), state={})

        expected = list(zip(block.attn_qs, block.mlp_qs))
        self.assertEqual(len(observed_pairs), 3)
        self.assertTrue(
            all(
                actual_attn is expected_attn and actual_mlp is expected_mlp
                for (actual_attn, actual_mlp), (expected_attn, expected_mlp) in zip(
                    observed_pairs, expected
                )
            )
        )

    def test_dual_forward_uses_matching_query_pairs_and_all_receive_gradients(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=4,
        )
        block.history_key_norm = IdentityNorm()
        block.norm = IdentityNorm()
        observed: list[torch.Tensor] = []

        def record_read(self, cache, partial_block, query, *args, **kwargs):
            del cache, args, kwargs
            observed.append(query)
            streams = self._to_stream_tensor(partial_block)
            hidden = streams.mean(dim=2) + query.sum()
            beta = torch.sigmoid(query.sum()).expand(*hidden.shape[:-1], 2)
            return hidden, streams[:, :, 0, :], streams[:, :, 1, :], beta

        block._dual_block_attn_res = MethodType(record_read, block)
        x = torch.randn(2, 2, 8, requires_grad=True)
        with mock.patch.multiple(
            rb,
            _DUAL_TRITON_HISTORY_APPEND_PACK=False,
            _DUAL_CACHE_HISTORY_PREFIX=False,
            _DUAL_CACHE_HISTORY_LOGITS=False,
            _DUAL_DEFER_STREAM_MIX=False,
            _DUAL_CURRENT_PARTIAL_RESIDUAL=False,
            _DUAL_ATTRES_LIMIT=False,
            _DUAL_ATTRES_LIKE_FIRST_ATTN=False,
            _DUAL_TE_ATTENTION_EPILOGUE=False,
            _RESIDUAL_TE_ATTENTION_EPILOGUE=False,
        ):
            output = block(x, state={})
        output.sum().backward()

        expected = [
            block.attn_qs[0],
            block.mlp_qs[0],
            block.attn_qs[1],
            block.mlp_qs[1],
            block.attn_qs[2],
            block.mlp_qs[2],
        ]
        self.assertEqual(len(observed), len(expected))
        self.assertTrue(all(actual is wanted for actual, wanted in zip(observed, expected)))
        self.assert_all_nonzero_finite_gradients(expected)

    def test_dual_paired_prefix_is_computed_once_per_atomic_layer(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=4,
        )
        block.history_key_norm = IdentityNorm()
        block.norm = IdentityNorm()
        observed_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_prefix(keys, values, attn_query, mlp_query, **kwargs):
            del keys, values, kwargs
            observed_pairs.append((attn_query, mlp_query))
            return None, None

        def record_read(self, cache, partial_block, query, *args, **kwargs):
            del cache, query, args, kwargs
            streams = self._to_stream_tensor(partial_block)
            hidden = streams.mean(dim=2)
            beta = hidden.new_ones((*hidden.shape[:-1], 2))
            return hidden, streams[:, :, 0, :], streams[:, :, 1, :], beta

        block._dual_block_attn_res = MethodType(record_read, block)
        with mock.patch.multiple(
            rb,
            _DUAL_TRITON_HISTORY_APPEND_PACK=False,
            _DUAL_CACHE_HISTORY_PREFIX=True,
            _DUAL_CACHE_HISTORY_PREFIX_PAIR=True,
            _DUAL_RECOMPUTE_HISTORY_VALUES=False,
            _DUAL_RECOMPUTE_HISTORY_KEYS=False,
            _DUAL_CACHE_HISTORY_LOGITS=False,
            _DUAL_DEFER_STREAM_MIX=False,
            _DUAL_CURRENT_PARTIAL_RESIDUAL=False,
            _DUAL_ATTRES_LIMIT=False,
            _DUAL_ATTRES_LIKE_FIRST_ATTN=False,
            _DUAL_TE_ATTENTION_EPILOGUE=False,
            _RESIDUAL_TE_ATTENTION_EPILOGUE=False,
            _dual_history_prefix_pair=record_prefix,
        ):
            block(torch.randn(2, 2, 8), state={})

        expected = list(zip(block.attn_qs, block.mlp_qs))
        self.assertEqual(len(observed_pairs), 3)
        self.assertTrue(
            all(
                actual_attn is expected_attn and actual_mlp is expected_mlp
                for (actual_attn, actual_mlp), (expected_attn, expected_mlp) in zip(
                    observed_pairs, expected
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
