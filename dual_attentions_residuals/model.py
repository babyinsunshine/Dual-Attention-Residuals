from __future__ import annotations

import importlib
import inspect
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .mcore_compat import call_with_supported_kwargs, first_tensor, require_megatron_core
from .residual_blocks import (
    AttentionResidualBlock,
    AttentionResidualOutputAggregation,
    DualAttentionResidualBlock,
    DualAttentionResidualOutputAggregation,
)


def _activation(name: str):
    if name.lower() in {"silu", "swiglu"}:
        return F.silu
    if name.lower() == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation_func: {name}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _debug_rank0() -> bool:
    if not _env_bool("MEGATRON_LOGIT_DEBUG", False):
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> str:
    with torch.no_grad():
        x = tensor.detach().float()
        return (
            f"{name}:shape={tuple(tensor.shape)} "
            f"mean={x.mean().item():.3e} rms={x.pow(2).mean().sqrt().item():.3e} "
            f"std={x.std(unbiased=False).item():.3e} max={x.abs().max().item():.3e}"
        )


def _param_debug_stats(name: str, module: nn.Module) -> Optional[str]:
    weight = getattr(module, "weight", None)
    if weight is None:
        try:
            weight = next(module.parameters())
        except StopIteration:
            return None
    if not isinstance(weight, torch.Tensor):
        return None
    return _tensor_debug_stats(name, weight)


def _moe_enabled(cfg: ExperimentConfig) -> bool:
    return cfg.model.num_moe_experts is not None and int(cfg.model.num_moe_experts) > 0


def _make_transformer_config(cfg: ExperimentConfig):
    require_megatron_core()
    from megatron.core.transformer.transformer_config import TransformerConfig

    m = cfg.model
    p = cfg.parallel
    attention_backend = m.attention_backend
    if isinstance(attention_backend, str):
        from megatron.core.transformer.enums import AttnBackend

        try:
            attention_backend = AttnBackend[attention_backend.lower()]
        except KeyError as exc:
            choices = ", ".join(backend.name for backend in AttnBackend)
            raise ValueError(f"Unknown attention_backend={m.attention_backend!r}; expected one of: {choices}") from exc
    kv_channels = m.kv_channels or (m.hidden_size // m.num_attention_heads)
    kwargs = {
        "num_layers": m.num_layers,
        "hidden_size": m.hidden_size,
        "num_attention_heads": m.num_attention_heads,
        "num_query_groups": m.num_query_groups,
        "kv_channels": kv_channels,
        "ffn_hidden_size": m.ffn_hidden_size,
        "hidden_dropout": m.hidden_dropout,
        "attention_dropout": m.attention_dropout,
        "attention_backend": attention_backend,
        "transformer_impl": m.transformer_impl,
        "init_method_std": m.init_method_std,
        "normalization": m.normalization,
        "layernorm_epsilon": m.layernorm_epsilon,
        "gated_linear_unit": m.gated_linear_unit,
        "activation_func": _activation(m.activation_func),
        "add_bias_linear": m.add_bias_linear,
        "add_qkv_bias": m.add_qkv_bias,
        "tensor_model_parallel_size": p.tensor_model_parallel_size,
        "pipeline_model_parallel_size": p.pipeline_model_parallel_size,
        "expert_model_parallel_size": p.expert_model_parallel_size,
        "context_parallel_size": p.context_parallel_size,
        "sequence_parallel": p.sequence_parallel,
        "recompute_granularity": cfg.train.recompute_granularity,
        "recompute_method": cfg.train.recompute_method,
        "recompute_num_layers": cfg.train.recompute_num_layers,
        "bf16": cfg.train.bf16,
        "fp16": cfg.train.fp16,
        "params_dtype": (
            torch.bfloat16
            if cfg.train.bf16
            else torch.float16
            if cfg.train.fp16
            else torch.float32
        ),
        "attention_softmax_in_fp32": True,
        "pipeline_dtype": (
            torch.bfloat16
            if cfg.train.bf16
            else torch.float16
            if cfg.train.fp16
            else torch.float32
        ),
    }
    if _moe_enabled(cfg):
        kwargs.update(
            {
                "num_moe_experts": m.num_moe_experts,
                "moe_router_topk": m.moe_router_topk,
                "moe_router_load_balancing_type": m.moe_router_load_balancing_type,
                "moe_aux_loss_coeff": m.moe_aux_loss_coeff,
                "moe_grouped_gemm": m.moe_grouped_gemm,
                "moe_permute_fusion": _env_bool("MEGATRON_MOE_PERMUTE_FUSION", m.moe_permute_fusion),
                "moe_router_fusion": _env_bool("MEGATRON_MOE_ROUTER_FUSION", m.moe_router_fusion),
                "moe_token_dispatcher_type": m.moe_token_dispatcher_type,
                "moe_expert_capacity_factor": m.moe_expert_capacity_factor,
            }
        )
    return call_with_supported_kwargs(TransformerConfig, **kwargs)


def _make_layer_spec(cfg: ExperimentConfig):
    require_megatron_core()
    if str(getattr(cfg.model, "transformer_impl", "")).lower() == "local":
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        return call_with_supported_kwargs(
            get_gpt_layer_local_spec,
            num_experts=cfg.model.num_moe_experts if _moe_enabled(cfg) else None,
            moe_grouped_gemm=cfg.model.moe_grouped_gemm if _moe_enabled(cfg) else False,
            qk_layernorm=False,
        )
    try:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

        return call_with_supported_kwargs(
            get_gpt_layer_with_transformer_engine_spec,
            num_experts=cfg.model.num_moe_experts if _moe_enabled(cfg) else None,
            moe_grouped_gemm=cfg.model.moe_grouped_gemm if _moe_enabled(cfg) else False,
            qk_layernorm=False,
            multi_latent_attention=False,
        )
    except Exception:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        return call_with_supported_kwargs(
            get_gpt_layer_local_spec,
            num_experts=cfg.model.num_moe_experts if _moe_enabled(cfg) else None,
            moe_grouped_gemm=cfg.model.moe_grouped_gemm if _moe_enabled(cfg) else False,
            qk_layernorm=False,
        )


def _pp_rank() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_pipeline_model_parallel_rank())
    except Exception:
        return 0


def _pp_world_size() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_pipeline_model_parallel_world_size())
    except Exception:
        return 1


def _is_pp_first_stage() -> bool:
    try:
        from megatron.core import parallel_state

        return bool(parallel_state.is_pipeline_first_stage())
    except Exception:
        return True


def _is_pp_last_stage() -> bool:
    try:
        from megatron.core import parallel_state

        return bool(parallel_state.is_pipeline_last_stage())
    except Exception:
        return True


def _tp_world_size() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _sequence_parallel_ops():
    errors = []
    for module_name in ("megatron.core.tensor_parallel.mappings", "megatron.core.tensor_parallel"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        scatter = getattr(module, "scatter_to_sequence_parallel_region", None)
        gather = getattr(module, "gather_from_sequence_parallel_region", None)
        if scatter is not None and gather is not None:
            return scatter, gather
        errors.append(f"{module_name}: missing scatter/gather sequence-parallel mappings")
    raise RuntimeError(
        "Megatron sequence parallel is enabled, but sequence-parallel mapping ops were not found. "
        + "; ".join(errors)
    )


def _stage_block_range(cfg: ExperimentConfig, pp_rank: int, pp_world_size: int) -> tuple[int, int, int, int]:
    block_layers = max(1, int(cfg.model.hyper_conn_block_layers))
    if cfg.model.num_layers % block_layers != 0:
        raise ValueError("num_layers must be divisible by hyper_conn_block_layers")
    num_blocks = cfg.model.num_layers // block_layers
    if bool(getattr(cfg.model, "attnres_pp_stage_aligned", True)) and num_blocks % pp_world_size != 0:
        raise ValueError(
            f"Block AttnRes PP alignment requires num_blocks ({num_blocks}) to be divisible by "
            f"pipeline_model_parallel_size ({pp_world_size})"
        )
    blocks_per_stage = num_blocks // pp_world_size
    start_block = pp_rank * blocks_per_stage
    end_block = (pp_rank + 1) * blocks_per_stage
    return start_block, end_block, start_block * block_layers, end_block * block_layers


def _make_rotary_embedding(cfg: ExperimentConfig, transformer_config) -> Optional[nn.Module]:
    if str(cfg.model.position_embedding_type).lower() != "rope":
        return None
    if bool(getattr(transformer_config, "multi_latent_attention", False)):
        return None
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    return call_with_supported_kwargs(
        RotaryEmbedding,
        kv_channels=transformer_config.kv_channels,
        rotary_percent=getattr(transformer_config, "rotary_percent", 1.0),
        rotary_interleaved=getattr(transformer_config, "rotary_interleaved", False),
        seq_len_interpolation_factor=getattr(transformer_config, "seq_len_interpolation_factor", None),
        rotary_base=cfg.model.rotary_base,
        rope_scaling=getattr(transformer_config, "rope_scaling", False),
        rope_scaling_factor=getattr(transformer_config, "rope_scaling_factor", 8.0),
        use_cpu_initialization=getattr(transformer_config, "use_cpu_initialization", False),
        cp_group=None,
    )


def _make_final_layernorm(transformer_config) -> nn.Module:
    from megatron.core.transformer.transformer_block import LayerNormImpl

    return call_with_supported_kwargs(
        LayerNormImpl,
        config=transformer_config,
        hidden_size=transformer_config.hidden_size,
        eps=transformer_config.layernorm_epsilon,
    )


class MCoreAtomicLayer(nn.Module):
    def __init__(self, transformer_config, layer_spec, layer_number: int, sequence_batch_size: int) -> None:
        super().__init__()
        require_megatron_core()
        from megatron.core.transformer.transformer_layer import TransformerLayer

        self.sequence_batch_size = max(1, int(sequence_batch_size))
        submodules = getattr(layer_spec, "submodules", layer_spec)
        self.layer = call_with_supported_kwargs(
            TransformerLayer,
            config=transformer_config,
            submodules=submodules,
            layer_number=layer_number,
        )
        self.last_aux_loss: Optional[torch.Tensor] = None
        self._mlp_apply = None
        self._mlp_accepts_padding_mask = False
        if hasattr(self.layer, "mlp"):
            from megatron.core.typed_torch import apply_module

            self._mlp_apply = apply_module(self.layer.mlp)
            try:
                self._mlp_accepts_padding_mask = "padding_mask" in inspect.signature(self._mlp_apply).parameters
            except (TypeError, ValueError):
                self._mlp_accepts_padding_mask = False

    def _to_sequence_batch(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        original_shape = tuple(hidden_states.shape)
        if hidden_states.ndim == 2:
            tokens, hidden = hidden_states.shape
            if tokens % self.sequence_batch_size != 0:
                raise ValueError(
                    f"Cannot restore sequence/batch dims from {original_shape} with batch={self.sequence_batch_size}"
                )
            hidden_states = hidden_states.view(tokens // self.sequence_batch_size, self.sequence_batch_size, hidden)
        return hidden_states, original_shape

    @staticmethod
    def _restore_shape(hidden_states: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 2 and hidden_states.ndim == 3:
            return hidden_states.reshape(original_shape)
        return hidden_states

    def attention_delta(self, hidden_states: torch.Tensor, *, attention_mask=None, rotary_pos_emb=None) -> torch.Tensor:
        self.last_aux_loss = None
        hidden_states, original_shape = self._to_sequence_batch(hidden_states)
        if hasattr(self.layer, "_forward_attention"):
            out = call_with_supported_kwargs(
                self.layer._forward_attention,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_context=None,
                packed_seq_params=None,
            )
            return self._restore_shape(first_tensor(out) - hidden_states, original_shape)
        out = self.layer(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        return self._restore_shape(first_tensor(out) - hidden_states, original_shape)

    def attention_raw_output(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
    ) -> torch.Tensor:
        """Run the TE attention branch without Megatron's residual add."""
        self.last_aux_loss = None
        hidden_states, original_shape = self._to_sequence_batch(hidden_states)
        if not hasattr(self.layer, "self_attention") or not hasattr(self.layer, "input_layernorm"):
            raise RuntimeError("raw attention output requires MCore TransformerLayer attention modules")
        if bool(getattr(self.layer, "recompute_input_layernorm", False)):
            raise RuntimeError("raw attention output does not support standalone input-layernorm recompute")
        dropout = float(getattr(self.layer, "hidden_dropout", 0.0))
        if dropout:
            raise RuntimeError("raw attention output requires hidden dropout to be disabled")
        normalized = self.layer.input_layernorm(hidden_states)
        out = self.layer.self_attention(
            normalized,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=None,
            packed_seq_params=None,
        )
        tensor = self._with_optional_bias(out)
        return self._restore_shape(tensor, original_shape)

    @staticmethod
    def _with_optional_bias(output) -> torch.Tensor:
        tensor = first_tensor(output)
        if isinstance(output, (tuple, list)) and len(output) > 1:
            bias = output[1]
            if hasattr(bias, "shape"):
                tensor = tensor + bias
        return tensor

    def mlp_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, original_shape = self._to_sequence_batch(hidden_states)
        if hasattr(self.layer, "_forward_pre_mlp_layernorm"):
            hidden_states = first_tensor(self.layer._forward_pre_mlp_layernorm(hidden_states))
        if self._mlp_apply is not None:
            if self._mlp_accepts_padding_mask:
                out = self._mlp_apply(hidden_states, padding_mask=None)
            else:
                out = self._mlp_apply(hidden_states)
            self.last_aux_loss = _extract_aux_loss(out)
            return self._restore_shape(self._with_optional_bias(out), original_shape)
        if hasattr(self.layer, "_forward_mlp"):
            out = call_with_supported_kwargs(self.layer._forward_mlp, hidden_states=hidden_states)
            tensor = first_tensor(out) - hidden_states
            self.last_aux_loss = _extract_aux_loss(out)
            return self._restore_shape(tensor, original_shape)
        out = self.layer(hidden_states)
        tensor = first_tensor(out)
        self.last_aux_loss = _extract_aux_loss(out)
        return self._restore_shape(tensor, original_shape)


def _extract_aux_loss(value) -> Optional[torch.Tensor]:
    if isinstance(value, dict):
        for key in ("moe_loss", "aux_loss", "moe_aux_loss"):
            item = value.get(key)
            if hasattr(item, "shape"):
                return item
    if isinstance(value, tuple):
        for item in value[1:]:
            if isinstance(item, dict):
                loss = _extract_aux_loss(item)
                if loss is not None:
                    return loss
            if hasattr(item, "shape") and item.ndim == 0:
                return item
    return None


class ResidualMCoreModel(nn.Module):
    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
    ) -> None:
        super().__init__()
        require_megatron_core()
        self.cfg = cfg
        self.variant = cfg.model.variant.lower()
        self.is_full_residual = self.variant in {"attention_residual", "dual_attention_residual"}
        self.is_dual_residual = self.variant in {"dual_attention_residual", "dual_attention_residual_block"}
        self.num_streams = 2 if self.is_dual_residual else 1
        self.transformer_config = _make_transformer_config(cfg)
        self.config = self.transformer_config
        self.layer_spec = _make_layer_spec(cfg)
        self.pp_rank = _pp_rank()
        self.pp_world_size = _pp_world_size()
        self.is_first_stage = _is_pp_first_stage() if pre_process is None else bool(pre_process)
        self.is_last_stage = _is_pp_last_stage() if post_process is None else bool(post_process)
        self.pre_process = self.is_first_stage
        self.post_process = self.is_last_stage
        # This flag is only true when we actually tie the local embedding and
        # output modules. Cross-stage tying is handled by GPTModel, but this
        # lightweight residual wrapper only supports same-stage tying.
        self.share_embeddings_and_output_weights = False
        self.position_embedding_type = cfg.model.position_embedding_type
        self.input_tensor = None
        self.residual_batch_size = int(cfg.train.micro_batch_size) * self.num_streams
        self.sequence_parallel = bool(cfg.parallel.sequence_parallel)
        self.tensor_model_parallel_size = max(int(cfg.parallel.tensor_model_parallel_size), _tp_world_size())
        self.num_blocks = cfg.model.num_layers // max(1, int(cfg.model.hyper_conn_block_layers))
        (
            self.stage_start_block,
            self.stage_end_block,
            self.stage_start_layer,
            self.stage_end_layer,
        ) = _stage_block_range(cfg, self.pp_rank, self.pp_world_size)
        self.embedding = _make_vocab_embedding(cfg, self.transformer_config) if self.is_first_stage else None
        self.output_layer = _make_output_layer(cfg, self.transformer_config) if self.is_last_stage else None
        self.final_norm = _make_final_layernorm(self.transformer_config) if self.is_last_stage else None
        if self.final_norm is not None:
            self.final_norm.to(dtype=self.transformer_config.params_dtype)
        self._setup_embedding_output_parameters()
        self.rotary_pos_emb = _make_rotary_embedding(cfg, self.transformer_config)
        block_layers = max(1, int(cfg.model.hyper_conn_block_layers))
        if cfg.model.num_layers % block_layers != 0:
            raise ValueError("num_layers must be divisible by hyper_conn_block_layers")
        atomic_layers = [
            MCoreAtomicLayer(
                self.transformer_config,
                self.layer_spec,
                layer_number=(i - self.stage_start_layer) + 1,
                sequence_batch_size=self.residual_batch_size,
            )
            for i in range(self.stage_start_layer, self.stage_end_layer)
        ]
        blocks = []
        for local_block_index, block_start in enumerate(range(self.stage_start_layer, self.stage_end_layer, block_layers)):
            layers = atomic_layers[local_block_index * block_layers : (local_block_index + 1) * block_layers]
            if self.variant == "attention_residual_block":
                blocks.append(AttentionResidualBlock(cfg, layers))
            elif self.variant == "dual_attention_residual_block":
                blocks.append(DualAttentionResidualBlock(cfg, layers, layer_index=block_start))
            else:
                raise ValueError(f"ResidualMCoreModel does not support variant={cfg.model.variant!r}")
        self.blocks = nn.ModuleList(blocks)
        if self.is_last_stage and self.variant in {"attention_residual_block", "dual_attention_residual_block"}:
            self.output_aggregation = (
                AttentionResidualOutputAggregation(cfg)
                if self.variant == "attention_residual_block"
                else DualAttentionResidualOutputAggregation(cfg)
            )
        else:
            self.output_aggregation = None

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, (list, tuple)) and len(input_tensor) == 1:
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return x.unsqueeze(2).expand(-1, -1, self.num_streams, -1).reshape(
            x.shape[0], x.shape[1] * self.num_streams, x.shape[2]
        ).contiguous()

    def _reduce_streams(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        seq, expanded_batch, hidden = x.shape
        return x.view(seq, expanded_batch // self.num_streams, self.num_streams, hidden).sum(dim=2)

    def _ensure_sequence_batch_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            return x
        tokens, hidden = x.shape
        if tokens % self.residual_batch_size != 0:
            raise ValueError(
                f"Cannot restore residual wrapper hidden shape from {tuple(x.shape)} "
                f"with batch={self.residual_batch_size}"
            )
        return x.view(tokens // self.residual_batch_size, self.residual_batch_size, hidden).contiguous()

    def _normalize_state_history(self, state: dict) -> None:
        history = state.get("history")
        if history is None:
            return
        if isinstance(history, torch.Tensor):
            state["history"] = [self._ensure_sequence_batch_hidden(history)]
            return
        state["history"] = [self._ensure_sequence_batch_hidden(item) for item in history]

    def _uses_sequence_parallel(self) -> bool:
        return self.sequence_parallel and self.tensor_model_parallel_size > 1

    def _scatter_sequence_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._uses_sequence_parallel():
            return hidden_states
        scatter, _ = _sequence_parallel_ops()
        return scatter(hidden_states)

    def _build_rotary_pos_emb(self, hidden_states: torch.Tensor):
        if self.rotary_pos_emb is None:
            return None
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None,
            None,
            hidden_states,
            self.transformer_config,
            None,
        )
        return self.rotary_pos_emb(rotary_seq_len)

    def _checkpoint_residual_block(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        state: dict,
        attention_mask: Optional[torch.Tensor],
        rotary_pos_emb,
    ) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        history = state.get("history")
        if history is None:
            history_inputs: tuple[torch.Tensor, ...] = ()
        elif isinstance(history, torch.Tensor):
            history_inputs = (history,)
        else:
            history_inputs = tuple(history)
        history_count = len(history_inputs)

        def block_forward(
            current: torch.Tensor,
            *previous_history: torch.Tensor,
            _block: nn.Module = block,
        ):
            local_state = {"history": list(previous_history)} if previous_history else {}
            output = _block(
                current,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                state=local_state,
            )
            updated_history = local_state.get("history", [])
            return (output, *updated_history[history_count:])

        outputs = checkpoint(
            block_forward,
            hidden_states,
            *history_inputs,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        state["history"] = [*history_inputs, *outputs[1:]]
        return outputs[0]

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.is_first_stage:
            if self.embedding is None:
                raise RuntimeError("first pipeline stage is missing token embedding")
            hidden_states = _call_embedding(self.embedding, tokens, position_ids)
            hidden_states = self._expand_streams(hidden_states)
            hidden_states = self._ensure_sequence_batch_hidden(hidden_states)
            state: dict = {}
        else:
            if self.input_tensor is None:
                raise RuntimeError("pipeline stage expected an input tensor from the previous stage")
            hidden_states, state = self._unpack_pp_input(self.input_tensor)
            hidden_states = self._ensure_sequence_batch_hidden(hidden_states)
        self._normalize_state_history(state)
        rotary_pos_emb = self._build_rotary_pos_emb(hidden_states)
        for block in self.blocks:
            if self.cfg.train.recompute_granularity == "full":
                hidden_states = self._checkpoint_residual_block(
                    block,
                    hidden_states,
                    state,
                    attention_mask,
                    rotary_pos_emb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    state=state,
                )
        if not self.is_last_stage:
            return self._pack_pp_output(hidden_states, state)
        if self.final_norm is None or self.output_layer is None:
            raise RuntimeError("last pipeline stage is missing output modules")
        debug_logits = _debug_rank0()
        if debug_logits:
            print(f"[logit_debug] variant={self.variant}", flush=True)
            embedding_weight = _embedding_weight(self.embedding)
            output_weight = getattr(self.output_layer, "weight", None)
            tied = (
                embedding_weight is not None
                and output_weight is not None
                and embedding_weight.data_ptr() == output_weight.data_ptr()
            )
            print(f"[logit_debug] embedding_output_tied={int(tied)}", flush=True)
            print("[logit_debug] " + _tensor_debug_stats("pre_output_aggregation", hidden_states), flush=True)
        if self.output_aggregation is not None:
            hidden_states = self.output_aggregation(hidden_states, state)
            if debug_logits:
                print("[logit_debug] " + _tensor_debug_stats("post_output_aggregation", hidden_states), flush=True)
        hidden_states = self._reduce_streams(hidden_states)
        if debug_logits:
            print("[logit_debug] " + _tensor_debug_stats("post_reduce_streams", hidden_states), flush=True)
        # Keep sequence-parallel shards here. Megatron's ColumnParallelLinear
        # owns the sequence all-gather when its config has sequence_parallel=True.
        hidden_states = hidden_states.to(dtype=self.transformer_config.params_dtype)
        hidden_states = self.final_norm(hidden_states).contiguous()
        if debug_logits:
            print("[logit_debug] " + _tensor_debug_stats("post_final_norm", hidden_states), flush=True)
            weight_stats = _param_debug_stats("output_weight", self.output_layer)
            if weight_stats is not None:
                print("[logit_debug] " + weight_stats, flush=True)
        logits = _call_output_layer(self.output_layer, hidden_states)
        if debug_logits:
            print("[logit_debug] " + _tensor_debug_stats("logits", logits), flush=True)
        if labels is not None:
            if debug_logits:
                try:
                    from megatron.core import parallel_state

                    tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
                    tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
                except Exception:
                    tp_rank = 0
                    tp_size = 1
                aligned_logits = _align_logits_to_labels(logits, labels)
                part_vocab = int(aligned_logits.shape[-1])
                vocab_start = tp_rank * part_vocab
                vocab_end = vocab_start + part_vocab
                target_mask = (labels >= vocab_start) & (labels < vocab_end)
                if bool(target_mask.any()):
                    local_labels = (labels[target_mask] - vocab_start).long()
                    local_logits = aligned_logits[target_mask]
                    target_logits = local_logits.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)
                    print(
                        "[logit_debug] "
                        + _tensor_debug_stats(
                            f"target_logits_local tp={tp_rank}/{tp_size} hit={int(target_mask.sum().item())}/{labels.numel()}",
                            target_logits,
                        ),
                        flush=True,
                    )
                    print(
                        "[logit_debug] "
                        + _tensor_debug_stats(f"local_lse tp={tp_rank}/{tp_size}", torch.logsumexp(local_logits.float(), dim=-1)),
                        flush=True,
                    )
                input_mask = (tokens >= vocab_start) & (tokens < vocab_end)
                if bool(input_mask.any()):
                    local_input_tokens = (tokens[input_mask] - vocab_start).long()
                    input_logits = aligned_logits[input_mask].gather(-1, local_input_tokens.unsqueeze(-1)).squeeze(-1)
                    print(
                        "[logit_debug] "
                        + _tensor_debug_stats(
                            f"input_token_logits_local tp={tp_rank}/{tp_size} hit={int(input_mask.sum().item())}/{tokens.numel()}",
                            input_logits,
                        ),
                        flush=True,
                    )
            losses = _parallel_cross_entropy(logits, labels)
            if debug_logits:
                print("[logit_debug] " + _tensor_debug_stats("token_losses", losses), flush=True)
                print(
                    "[logit_debug] "
                    + f"labels:shape={tuple(labels.shape)} min={int(labels.min().item())} max={int(labels.max().item())}",
                    flush=True,
                )
            return losses
        return logits

    def _history_from_state(self, state: dict) -> list[torch.Tensor]:
        history = state.get("history")
        if history is None:
            return []
        if isinstance(history, torch.Tensor):
            return [self._ensure_sequence_batch_hidden(history)]
        return [self._ensure_sequence_batch_hidden(item) for item in history]

    def _pack_pp_output(self, hidden_states: torch.Tensor, state: dict) -> torch.Tensor:
        history = self._history_from_state(state)
        expected_history = self.stage_end_block
        if len(history) != expected_history:
            raise RuntimeError(
                "Residual PP package history length mismatch: "
                f"got {len(history)}, expected {expected_history} after block {self.stage_end_block}"
            )
        tensors = [hidden_states, *history]
        return torch.stack(tensors, dim=0).contiguous()

    def _unpack_pp_input(self, package) -> tuple[torch.Tensor, dict]:
        if isinstance(package, (list, tuple)) and len(package) == 1:
            package = package[0]
        if not isinstance(package, torch.Tensor) or package.ndim != 4:
            raise RuntimeError(
                "Residual PP input must be a packed tensor [slot, seq, batch, hidden]; "
                f"got {type(package).__name__} with shape={getattr(package, 'shape', None)}"
            )
        expected_slots = self.stage_start_block + 1
        if int(package.shape[0]) != expected_slots:
            raise RuntimeError(
                f"Residual PP input slot mismatch: got {tuple(package.shape)}, expected slots={expected_slots}"
            )
        hidden_states = self._ensure_sequence_batch_hidden(package[0].contiguous())
        history = [
            self._ensure_sequence_batch_hidden(package[i + 1].contiguous())
            for i in range(self.stage_start_block)
        ]
        return hidden_states, {"history": history}

    def _setup_embedding_output_parameters(self) -> None:
        embedding_weight = _embedding_weight(self.embedding)
        output_weight = getattr(self.output_layer, "weight", None) if self.output_layer is not None else None
        share_requested = bool(getattr(self.cfg.model, "share_embeddings_and_output_weights", False))
        embedding_init_std = getattr(self.cfg.model, "embedding_init_method_std", None)
        output_init_std = getattr(self.cfg.model, "output_init_method_std", None)

        with torch.no_grad():
            if embedding_weight is not None and embedding_init_std is not None:
                torch.nn.init.normal_(embedding_weight, mean=0.0, std=float(embedding_init_std))
            if (
                output_weight is not None
                and output_init_std is not None
                and not share_requested
            ):
                torch.nn.init.normal_(output_weight, mean=0.0, std=float(output_init_std))

        if share_requested:
            if embedding_weight is not None and self.output_layer is not None:
                self.output_layer.weight = embedding_weight
                output_weight = embedding_weight
                self.share_embeddings_and_output_weights = True
            elif self.is_first_stage or self.is_last_stage:
                raise ValueError(
                    "ResidualMCoreModel can only share embedding/output weights when "
                    "the first and last pipeline stages are the same rank. Use PP=1, "
                    "or set model.share_embeddings_and_output_weights: false for residual PP runs."
                )

        for param in (embedding_weight, output_weight):
            if param is None:
                continue
            param.is_embedding_or_output_parameter = True
            if bool(getattr(self.transformer_config, "use_mup", False)):
                param.is_embedding_parameter = True


def _make_vocab_embedding(cfg: ExperimentConfig, transformer_config) -> nn.Module:
    try:
        from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

        return call_with_supported_kwargs(
            LanguageModelEmbedding,
            config=transformer_config,
            vocab_size=cfg.model.vocab_size,
            max_sequence_length=cfg.model.seq_length,
            position_embedding_type=cfg.model.position_embedding_type,
            scatter_to_sequence_parallel=True,
        )
    except Exception:
        from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

        emb = call_with_supported_kwargs(
            VocabParallelEmbedding,
            num_embeddings=cfg.model.vocab_size,
            embedding_dim=cfg.model.hidden_size,
            config=transformer_config,
            init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=cfg.model.init_method_std),
        )
        setattr(emb, "_dar_raw_vocab_embedding", True)
        return emb


def _make_output_layer(cfg: ExperimentConfig, transformer_config) -> nn.Module:
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        return call_with_supported_kwargs(
            ColumnParallelLinear,
            input_size=cfg.model.hidden_size,
            output_size=cfg.model.vocab_size,
            config=transformer_config,
            init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=cfg.model.init_method_std),
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
    except Exception:
        head = nn.Linear(cfg.model.hidden_size, cfg.model.vocab_size, bias=False)
        nn.init.normal_(head.weight, mean=0.0, std=cfg.model.init_method_std)
        return head


def _call_output_layer(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    out = layer(hidden_states)
    return first_tensor(out)


def _call_embedding(embedding: nn.Module, tokens: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
    if hasattr(embedding, "word_embeddings"):
        return embedding(input_ids=tokens, position_ids=position_ids)
    out = embedding(tokens)
    if out.ndim == 3 and out.shape[0] == tokens.shape[0]:
        out = out.transpose(0, 1).contiguous()
    return out


def _embedding_weight(embedding: Optional[nn.Module]) -> Optional[torch.nn.Parameter]:
    if embedding is None:
        return None
    word_embeddings = getattr(embedding, "word_embeddings", None)
    if word_embeddings is not None and getattr(word_embeddings, "weight", None) is not None:
        return word_embeddings.weight
    weight = getattr(embedding, "weight", None)
    return weight if isinstance(weight, torch.nn.Parameter) else None


def _build_official_gpt_model(
    cfg: ExperimentConfig,
    *,
    pre_process: Optional[bool] = None,
    post_process: Optional[bool] = None,
) -> nn.Module:
    require_megatron_core()
    from megatron.core.models.gpt.gpt_model import GPTModel

    is_first_stage = _is_pp_first_stage() if pre_process is None else bool(pre_process)
    is_last_stage = _is_pp_last_stage() if post_process is None else bool(post_process)
    transformer_config = _make_transformer_config(cfg)
    layer_spec = _make_layer_spec(cfg)
    model = call_with_supported_kwargs(
        GPTModel,
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=cfg.model.vocab_size,
        max_sequence_length=cfg.model.seq_length,
        pre_process=is_first_stage,
        post_process=is_last_stage,
        fp16_lm_cross_entropy=cfg.train.fp16,
        parallel_output=True,
        share_embeddings_and_output_weights=cfg.model.share_embeddings_and_output_weights,
        position_embedding_type=cfg.model.position_embedding_type,
        rotary_base=cfg.model.rotary_base,
    )
    if not hasattr(model, "transformer_config"):
        model.transformer_config = transformer_config
    return model


def _align_logits_to_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        return logits
    if tuple(logits.shape[:2]) == tuple(labels.shape):
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.transpose(0, 1).contiguous()
    return logits


def _prepare_parallel_ce_inputs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align logits and labels without allowing broadcasted CE shapes."""
    if logits.ndim != 3 or labels.ndim != 2:
        return logits, labels.long()
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        logits = _align_logits_to_labels(logits, labels)
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise RuntimeError(
            "Cannot align logits and labels for vocab-parallel CE: "
            f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    return logits.contiguous(), labels.contiguous().long()


def _parallel_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = _align_logits_to_labels(logits, labels)
    from megatron.core import tensor_parallel

    ce_logits, ce_labels = _prepare_parallel_ce_inputs(logits, labels)
    return tensor_parallel.vocab_parallel_cross_entropy(ce_logits.float(), ce_labels)


def build_model(
    cfg: ExperimentConfig,
    *,
    pre_process: Optional[bool] = None,
    post_process: Optional[bool] = None,
) -> nn.Module:
    variant = cfg.model.variant.lower()
    if variant == "baseline":
        return _build_official_gpt_model(cfg, pre_process=pre_process, post_process=post_process)
    if variant in {"attention_residual_block", "dual_attention_residual_block"}:
        return ResidualMCoreModel(cfg, pre_process=pre_process, post_process=post_process)
    raise ValueError(f"Unsupported Megatron variant: {cfg.model.variant}")
