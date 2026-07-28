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


class ScaledNorm(nn.Module):
    force_fp32 = False

    def __init__(self, scale: float, eps: float) -> None:
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


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

    def test_dual_owns_complete_distinct_attention_and_mlp_state_per_layer(self) -> None:
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

        states = [*block.attn_states, *block.mlp_states]
        query_pairs = [state.q for state in states]
        query_rows = [query[row] for query in query_pairs for row in range(2)]
        self.assertEqual(len(block.attn_states), 3)
        self.assertEqual(len(block.mlp_states), 3)
        self.assertEqual(len({id(state) for state in states}), 6)
        self.assertEqual([tuple(query.shape) for query in query_pairs], [(2, 8)] * 6)
        self.assertEqual(len({row.data_ptr() for row in query_rows}), 12)
        parameter_names = (
            "q",
            "static_alpha",
            "static_beta",
            "dynamic_ab_fn",
            "dynamic_residual_mix_fn",
            "static_residual_logit",
            "pre_branch_scale",
            "h_post_scale",
            "residual_scale",
        )
        for parameter_name in parameter_names:
            parameters = [getattr(state, parameter_name) for state in states]
            self.assertEqual(len({id(parameter) for parameter in parameters}), 6)
            self.assertEqual(len({parameter.data_ptr() for parameter in parameters}), 6)
            self.assertTrue(all(parameter.sequence_parallel for parameter in parameters))
        self.assertEqual(len({id(state.norm) for state in states}), 6)
        self.assertEqual(len({id(state.history_key_norm) for state in states}), 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual(Counter(calls), Counter({1: 4, 0: 2}))

        state_keys = set(block.state_dict())
        for branch in ("attn_states", "mlp_states"):
            for index in range(3):
                for parameter_name in parameter_names:
                    self.assertIn(f"{branch}.{index}.{parameter_name}", state_keys)
        self.assertNotIn("attn_q", state_keys)
        self.assertNotIn("mlp_q", state_keys)
        self.assertNotIn("attn_static_alpha", state_keys)
        self.assertNotIn("mlp_static_alpha", state_keys)


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
        for residual_state in (*block.attn_states, *block.mlp_states):
            residual_state.history_key_norm = IdentityNorm()
        observed: list[nn.Module] = []

        def record_read(self, cache, partial_block, residual_state, **kwargs):
            del cache, kwargs
            observed.append(residual_state)
            streams = self._to_stream_tensor(partial_block)
            parameter_sum = sum(parameter.sum() for parameter in residual_state.parameters())
            hidden = streams.mean(dim=2) + parameter_sum
            beta = torch.sigmoid(parameter_sum).expand(*hidden.shape[:-1], 2)
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
            block.attn_states[0],
            block.mlp_states[0],
            block.attn_states[1],
            block.mlp_states[1],
            block.attn_states[2],
            block.mlp_states[2],
        ]
        self.assertEqual(len(observed), len(expected))
        self.assertTrue(all(actual is wanted for actual, wanted in zip(observed, expected)))
        parameters = [parameter for residual_state in expected for parameter in residual_state.parameters()]
        self.assert_all_nonzero_finite_gradients(parameters)

    def test_dual_paired_prefix_is_computed_once_per_atomic_layer(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=4,
        )
        for attn_state, mlp_state in zip(block.attn_states, block.mlp_states):
            shared_test_norm = IdentityNorm()
            attn_state.history_key_norm = shared_test_norm
            mlp_state.history_key_norm = shared_test_norm
            attn_state.norm = IdentityNorm()
            mlp_state.norm = IdentityNorm()
        observed_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_prefix(keys, values, attn_query, mlp_query, **kwargs):
            del keys, values, kwargs
            observed_pairs.append((attn_query, mlp_query))
            return None, None

        def record_read(self, cache, partial_block, residual_state, **kwargs):
            del cache, residual_state, kwargs
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

        expected = [(attn_state.q, mlp_state.q) for attn_state, mlp_state in zip(block.attn_states, block.mlp_states)]
        self.assertEqual(len(observed_pairs), 3)
        self.assertTrue(
            all(
                actual_attn is expected_attn and actual_mlp is expected_mlp
                for (actual_attn, actual_mlp), (expected_attn, expected_mlp) in zip(
                    observed_pairs, expected
                )
            )
        )

    def test_dual_history_key_cache_uses_the_selected_sublayer_norm(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=0,
        )
        first_norm = ScaledNorm(scale=1.0, eps=1.0e-5)
        different_norm = ScaledNorm(scale=2.0, eps=1.0e-5)
        for residual_state in (*block.attn_states, *block.mlp_states):
            residual_state.history_key_norm = first_norm
        block.attn_states[0].history_key_norm = first_norm
        block.attn_states[1].history_key_norm = different_norm

        with mock.patch.object(rb, "_DUAL_TRITON_HISTORY_APPEND_PACK", False):
            cache = block._get_cache(torch.randn(2, 2, 8), {})
            first_keys = block._history_keys(cache, first_norm)
            different_keys = block._history_keys(cache, different_norm)

        self.assertEqual(len(cache["keys_by_norm"]), 2)
        self.assertIsNot(first_keys, different_keys)
        self.assertTrue(torch.equal(different_keys, first_keys * 2.0))

    def test_dual_paired_prefix_is_disabled_when_branch_norms_differ(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=0,
        )
        for attn_state, mlp_state in zip(block.attn_states, block.mlp_states):
            attn_state.history_key_norm = ScaledNorm(scale=1.0, eps=1.0e-5)
            mlp_state.history_key_norm = ScaledNorm(scale=2.0, eps=1.0e-5)
        paired_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        single_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_pair(keys, values, *args, **kwargs):
            del values, args, kwargs
            paired_calls.append((keys, keys))
            return None, None

        def record_single(keys, values, query):
            del values
            single_calls.append((keys, query))
            return None

        def record_read(self, cache, partial_block, residual_state, **kwargs):
            del cache, residual_state, kwargs
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
            _dual_history_prefix_pair=record_pair,
            _dual_history_prefix=record_single,
        ):
            block(torch.randn(2, 2, 8), state={})

        self.assertEqual(paired_calls, [])
        self.assertEqual(len(single_calls), 5)
        keys_by_query = {id(query): keys for keys, query in single_calls}
        self.assertNotIn(id(block.attn_states[0].q), keys_by_query)
        self.assertIn(id(block.mlp_states[0].q), keys_by_query)
        for layer_index in (1, 2):
            attn_keys = keys_by_query[id(block.attn_states[layer_index].q)]
            mlp_keys = keys_by_query[id(block.mlp_states[layer_index].q)]
            self.assertTrue(torch.equal(mlp_keys, attn_keys * 2.0))

    def test_dual_output_aggregation_recomputes_keys_for_its_own_norm(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=0,
        )
        block_norm = ScaledNorm(scale=2.0, eps=1.0e-5)
        aggregation_norm = ScaledNorm(scale=3.0, eps=1.0e-5)
        for residual_state in (*block.attn_states, *block.mlp_states):
            residual_state.history_key_norm = block_norm
        raw = torch.randn(2, 2, 8)
        completed = torch.randn(2, 2, 8)
        state = {"history": [raw, completed]}
        with mock.patch.object(rb, "_DUAL_TRITON_HISTORY_APPEND_PACK", False):
            cache = block._get_cache(raw, state)
            block._history_keys(cache, block_norm)

        aggregation = rb.DualAttentionResidualOutputAggregation(
            make_config("dual_attention_residual_block")
        )
        aggregation.history_key_norm = aggregation_norm
        final_hidden = torch.randn(2, 2, 8)
        packed_keys, _packed_values, _current_key, _current_value = (
            aggregation._pack_final_history(final_hidden, state)
        )
        expected_completed_key, _ = aggregation._pack(completed)

        self.assertTrue(torch.equal(packed_keys[:, 0], expected_completed_key))

    def test_dual_embed_as_partial_initializes_empty_triton_append_inputs(self) -> None:
        block = rb.DualAttentionResidualBlock(
            make_config("dual_attention_residual_block"),
            make_layers(),
            layer_index=0,
        )
        x = torch.randn(2, 2, 8)
        observed: list[tuple[torch.Tensor, torch.Tensor]] = []

        def record_append(previous_keys, previous_values, stream_value, *args):
            del args
            observed.append((previous_keys, previous_values))
            self.assertEqual(tuple(previous_keys.shape), (2, 0, 1, 2, 8))
            self.assertEqual(tuple(previous_values.shape), (2, 0, 1, 2, 8))
            keys = stream_value.transpose(0, 2).contiguous().unsqueeze(1)
            values = block._history_value(stream_value).unsqueeze(1)
            return keys, values

        with mock.patch.multiple(
            rb,
            _DUAL_EMBED_AS_PARTIAL=True,
            _DUAL_TRITON_HISTORY_APPEND_PACK=True,
            _DUAL_TRITON_HISTORY_APPEND_PACK_STRICT=True,
            _dual_residual_triton_is_available=lambda: True,
            triton_dual_history_append_pack=record_append,
        ):
            cache = block._get_cache(x, {})
            block._history_keys(cache, block.attn_states[0].history_key_norm)
            block._append_block(cache, x)

        self.assertEqual(len(observed), 1)


if __name__ == "__main__":
    unittest.main()
