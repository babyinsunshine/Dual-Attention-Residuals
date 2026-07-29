from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    variant: str = "baseline"
    seq_length: int = 4096
    vocab_size: int = 131072
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_layers: int = 32
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: Optional[int] = None
    init_method_std: float = 0.006
    embedding_init_method_std: Optional[float] = None
    output_init_method_std: Optional[float] = None
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_backend: Optional[Any] = None
    transformer_impl: Optional[str] = None
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1.0e-5
    gated_linear_unit: bool = True
    activation_func: str = "silu"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "rope"
    rotary_base: float = 1000000.0
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    num_moe_experts: Optional[int] = 8
    moe_router_topk: Optional[int] = 2
    moe_router_load_balancing_type: Optional[str] = "aux_loss"
    moe_aux_loss_coeff: Optional[float] = 1.0e-2
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = False
    moe_router_fusion: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_expert_capacity_factor: Optional[float] = None
    hyper_conn_block_layers: int = 4
    hyper_conn_n: int = 2
    hyper_conn_attres_separate_k_proj: bool = False
    attnres_pp_stage_aligned: bool = True
    attnres_pp_send_history: bool = True
    dual_residual_mix: str = "scalar_mhc_2x2"
    dual_kv_mode: str = "cross"


@dataclass
class ParallelConfig:
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 2
    expert_model_parallel_size: int = 4
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    virtual_pipeline_model_parallel_size: Optional[int] = None


@dataclass
class TrainConfig:
    output_dir: str = "outputs/megatron_7b_moe"
    train_tokens_file: str = "data/fineweb_edu_100BT_pretok/train.bin"
    val_tokens_file: str = "data/fineweb_edu_100BT_pretok/val.bin"
    seed: int = 42
    micro_batch_size: int = 4
    global_batch_size: int = 128
    train_iters: int = 100000
    eval_interval: int = 1000
    eval_iters: int = 20
    save_interval: int = 1000
    log_interval: int = 1
    shuffle_buffer_size: int = 10000
    start_token: int = 0
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    clip_grad: float = 1.0
    bf16: bool = True
    fp16: bool = False
    recompute_granularity: Optional[str] = "selective"
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    use_distributed_optimizer: bool = True
    use_megatron_fsdp: bool = False
    data_parallel_sharding_strategy: str = "optim_grads_params"
    ckpt_format: str = "torch_dist"
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    dataloader_workers: int = 2


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _asdict_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": cfg.model.__dict__,
        "parallel": cfg.parallel.__dict__,
        "train": cfg.train.__dict__,
    }


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    merged = _merge_dict(_asdict_config(ExperimentConfig()), raw_cfg)
    cfg = ExperimentConfig(
        model=ModelConfig(**merged["model"]),
        parallel=ParallelConfig(**merged["parallel"]),
        train=TrainConfig(**merged["train"]),
    )
    root = config_path.parent.parent
    if not Path(cfg.train.output_dir).is_absolute():
        cfg.train.output_dir = str((root / cfg.train.output_dir).resolve())
    if not Path(cfg.train.train_tokens_file).is_absolute():
        cfg.train.train_tokens_file = str((root / cfg.train.train_tokens_file).resolve())
    if not Path(cfg.train.val_tokens_file).is_absolute():
        cfg.train.val_tokens_file = str((root / cfg.train.val_tokens_file).resolve())
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    supported_variants = {
        "baseline",
        "attention_residual_block",
        "dual_attention_residual_block",
    }
    if cfg.model.variant not in supported_variants:
        choices = ", ".join(sorted(supported_variants))
        raise ValueError(f"supported variants are: {choices}; got {cfg.model.variant!r}")
    if isinstance(cfg.train.recompute_granularity, str):
        if cfg.train.recompute_granularity.lower() in {"", "none", "null", "off", "false", "0"}:
            cfg.train.recompute_granularity = None
    block_layers = max(1, int(cfg.model.hyper_conn_block_layers))
    full_variants: set[str] = set()
    block_variants = {"attention_residual_block", "dual_attention_residual_block"}
    residual_variants = full_variants | block_variants
    if cfg.model.variant in full_variants and block_layers != 1:
        raise ValueError(f"{cfg.model.variant} requires model.hyper_conn_block_layers=1")
    if cfg.model.variant in full_variants and cfg.parallel.pipeline_model_parallel_size != 1:
        raise ValueError(f"{cfg.model.variant} currently supports pipeline_model_parallel_size=1 only")
    if cfg.model.num_layers % block_layers != 0:
        raise ValueError(
            f"model.num_layers ({cfg.model.num_layers}) must be divisible by "
            f"model.hyper_conn_block_layers ({block_layers})"
        )
    num_blocks = cfg.model.num_layers // block_layers
    if cfg.model.variant in block_variants:
        if cfg.model.attnres_pp_stage_aligned and num_blocks % cfg.parallel.pipeline_model_parallel_size != 0:
            raise ValueError(
                f"Block AttnRes PP alignment requires num_blocks ({num_blocks}) to be divisible by "
                f"pipeline_model_parallel_size ({cfg.parallel.pipeline_model_parallel_size})"
            )
        if cfg.parallel.pipeline_model_parallel_size > 1 and not cfg.model.attnres_pp_send_history:
            raise ValueError("Block AttnRes PP requires model.attnres_pp_send_history=true")
    if cfg.model.variant in {"dual_attention_residual", "dual_attention_residual_block"}:
        if cfg.model.dual_kv_mode.lower() not in {"cross", "cross_kv"}:
            raise ValueError(f"Megatron {cfg.model.variant} currently supports only dual_kv_mode='cross'")
        if cfg.model.dual_residual_mix.lower() != "scalar_mhc_2x2":
            raise ValueError(f"Megatron {cfg.model.variant} currently supports only scalar_mhc_2x2 mixing")
