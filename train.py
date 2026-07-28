from __future__ import annotations

import argparse
import atexit
import gc
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path


def _init_rank_local_torch_cache() -> None:
    if os.environ.get("MEGATRON_DISABLE_RANK_LOCAL_TORCH_CACHE", "0").lower() in {"1", "true", "yes", "on"}:
        return
    local_rank = os.environ.get("LOCAL_RANK", "0")
    global_rank = os.environ.get("RANK", local_rank)
    node_rank = os.environ.get("NODE_RANK", "0")
    base = os.environ.get("MEGATRON_TORCH_CACHE_BASE")
    if not base:
        existing = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if existing:
            existing_path = Path(existing)
            base = str(existing_path.parent if existing_path.name == "inductor" else existing_path)
        else:
            base = f".cache/torch/node_{node_rank}"
    rank_root = Path(base) / f"rank_{global_rank}_local_{local_rank}"
    env_dirs = {
        "TORCHINDUCTOR_CACHE_DIR": rank_root / "inductor",
        "TRITON_CACHE_DIR": rank_root / "triton",
        "TORCH_EXTENSIONS_DIR": rank_root / "extensions",
    }
    for name, path in env_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tmp_path = Path(os.environ.get("MEGATRON_SHORT_TMPDIR", f"/tmp/tc_{global_rank}_{local_rank}"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_path)


_init_rank_local_torch_cache()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dual_attentions_residuals import build_model, load_config
from dual_attentions_residuals.data import PackedTokenDataset, batch_to_tokens_and_labels
from dual_attentions_residuals.mcore_compat import call_with_supported_kwargs, require_megatron_core


_WANDB_RUN = None


def _ensure_torch_rms_norm() -> None:
    if hasattr(F, "rms_norm"):
        return

    def _rms_norm(input, normalized_shape, weight=None, eps=None):
        if isinstance(normalized_shape, int):
            normalized_shape_tuple = (normalized_shape,)
        else:
            normalized_shape_tuple = tuple(normalized_shape)
        dims = tuple(range(input.dim() - len(normalized_shape_tuple), input.dim()))
        if eps is None:
            eps = torch.finfo(input.dtype).eps
        input_float = input.float()
        variance = input_float.pow(2).mean(dim=dims, keepdim=True)
        output = input_float * torch.rsqrt(variance + eps)
        output = output.to(dtype=input.dtype)
        if weight is not None:
            output = output * weight
        return output

    F.rms_norm = _rms_norm


_ensure_torch_rms_norm()


class PipelineDDP(torch.nn.Module):
    """Torch DDP wrapper that keeps Megatron pipeline hooks visible."""

    def __init__(self, module: torch.nn.Module, local_rank: int, process_group) -> None:
        super().__init__()
        self.ddp = torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            process_group=process_group,
            find_unused_parameters=False,
        )

    @property
    def module(self) -> torch.nn.Module:
        return self.ddp.module

    @property
    def config(self):
        return getattr(self.ddp.module, "config", getattr(self.ddp.module, "transformer_config", None))

    @property
    def transformer_config(self):
        return getattr(self.ddp.module, "transformer_config", self.config)

    @property
    def cfg(self):
        return getattr(self.ddp.module, "cfg", None)

    def set_input_tensor(self, input_tensor) -> None:
        if hasattr(self.ddp.module, "set_input_tensor"):
            self.ddp.module.set_input_tensor(input_tensor)

    def no_sync(self):
        return self.ddp.no_sync()

    def forward(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)


class MegatronDDPProxy(torch.nn.Module):
    """Megatron-Core DDP wrapper with the small interface our launcher expects."""

    def __init__(self, ddp: torch.nn.Module) -> None:
        super().__init__()
        self.ddp = ddp

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            ddp = super().__getattr__("ddp")
            return getattr(ddp, name)

    @property
    def module(self) -> torch.nn.Module:
        return self.ddp.module

    @property
    def config(self):
        return getattr(self.ddp, "config", getattr(self.ddp.module, "transformer_config", None))

    @property
    def transformer_config(self):
        return getattr(self.ddp.module, "transformer_config", self.config)

    @property
    def cfg(self):
        return getattr(self.ddp.module, "cfg", None)

    def set_input_tensor(self, input_tensor) -> None:
        if hasattr(self.ddp, "set_input_tensor"):
            self.ddp.set_input_tensor(input_tensor)
        elif hasattr(self.ddp.module, "set_input_tensor"):
            self.ddp.module.set_input_tensor(input_tensor)

    def no_sync(self):
        return self.ddp.no_sync()

    def forward(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron-Core 7B MoE residual training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--recompute-granularity", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-tokens-file", type=str, default=None)
    parser.add_argument("--val-tokens-file", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    return parser.parse_args()


def init_distributed(cfg) -> int:
    require_megatron_core()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    world_size = torch.distributed.get_world_size()
    shard_size = (
        cfg.parallel.tensor_model_parallel_size
        * cfg.parallel.pipeline_model_parallel_size
        * cfg.parallel.expert_model_parallel_size
        * cfg.parallel.context_parallel_size
    )
    if world_size % shard_size != 0:
        raise ValueError(f"world_size={world_size} is not divisible by TP*PP*EP*CP={shard_size}")
    from megatron.core import parallel_state

    call_with_supported_kwargs(
        parallel_state.initialize_model_parallel,
        tensor_model_parallel_size=cfg.parallel.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.parallel.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=cfg.parallel.virtual_pipeline_model_parallel_size,
        context_parallel_size=cfg.parallel.context_parallel_size,
        expert_model_parallel_size=cfg.parallel.expert_model_parallel_size,
    )
    if torch.cuda.is_available():
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        # GPTModel initializes tensor-parallel weights under this RNG tracker.
        # Residual custom models did not hit it, but baseline requires it.
        model_parallel_cuda_manual_seed(cfg.train.seed)
    return local_rank


def data_parallel_world_size() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_data_parallel_world_size())
    except Exception:
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def data_parallel_group():
    try:
        from megatron.core import parallel_state

        return parallel_state.get_data_parallel_group()
    except Exception:
        return None


def data_parallel_rank() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_data_parallel_rank())
    except Exception:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def pipeline_world_size() -> int:
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_pipeline_model_parallel_world_size())
    except Exception:
        return 1


def is_pipeline_last_stage() -> bool:
    try:
        from megatron.core import parallel_state

        return bool(parallel_state.is_pipeline_last_stage())
    except Exception:
        return True


def should_log_from_rank() -> bool:
    if not is_pipeline_last_stage() or data_parallel_rank() != 0:
        return False
    try:
        from megatron.core import parallel_state

        if int(parallel_state.get_tensor_model_parallel_rank()) != 0:
            return False
        if hasattr(parallel_state, "get_expert_model_parallel_rank"):
            if int(parallel_state.get_expert_model_parallel_rank()) != 0:
                return False
    except Exception:
        pass
    return True


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    if not torch.distributed.is_initialized():
        return model
    group = data_parallel_group()
    return PipelineDDP(model, local_rank, group)


def _is_custom_residual_variant(cfg) -> bool:
    return str(getattr(cfg.model, "variant", "")).lower() in {
        "attention_residual_block",
        "dual_attention_residual_block",
    }


def prefer_mcore_training_stack(cfg=None) -> bool:
    value = os.environ.get("MEGATRON_USE_MCORE_TRAINING")
    if value is not None and value != "":
        return value.lower() not in {"0", "false", "no"}
    return True


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def get_model_transformer_config(model: torch.nn.Module):
    for attr in ("transformer_config", "config"):
        value = getattr(model, attr, None)
        if value is not None:
            return value
    raise RuntimeError(
        f"{type(model).__name__} does not expose Megatron TransformerConfig "
        "via .transformer_config or .config"
    )


def wrap_megatron_ddp(model: torch.nn.Module, cfg) -> torch.nn.Module:
    from megatron.core.distributed import (
        DistributedDataParallel as MCoreDistributedDataParallel,
        DistributedDataParallelConfig,
    )

    overlap_grad_reduce = env_bool(
        "MEGATRON_OVERLAP_GRAD_REDUCE",
        bool(getattr(cfg.train, "overlap_grad_reduce", True)),
    )
    overlap_param_gather = env_bool(
        "MEGATRON_OVERLAP_PARAM_GATHER",
        bool(getattr(cfg.train, "overlap_param_gather", False)),
    )
    ddp_config = call_with_supported_kwargs(
        DistributedDataParallelConfig,
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        use_distributed_optimizer=bool(cfg.train.use_distributed_optimizer),
        average_in_collective=False,
    )
    # Some Megatron-Core versions expose a class signature that inspect.signature()
    # cannot reliably filter through.  Use the canonical positional constructor
    # so the required TransformerConfig is never dropped.
    ddp = MCoreDistributedDataParallel(
        get_model_transformer_config(model),
        ddp_config,
        model,
        disable_bucketing=False,
    )
    wrapped = MegatronDDPProxy(ddp)
    setattr(wrapped, "_dar_training_stack", "mcore_ddp")
    setattr(wrapped, "_dar_overlap_grad_reduce", overlap_grad_reduce)
    setattr(wrapped, "_dar_overlap_param_gather", overlap_param_gather)
    return wrapped


def is_mcore_ddp_model(model: torch.nn.Module) -> bool:
    return getattr(model, "_dar_training_stack", "") == "mcore_ddp"


def optimizer_model_chunk(model: torch.nn.Module) -> torch.nn.Module:
    """Return the native model chunk object expected by Megatron-Core optimizers."""
    if isinstance(model, MegatronDDPProxy):
        return model.ddp
    return model


def build_megatron_optimizer(model: torch.nn.Module, cfg):
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    params_dtype = (
        torch.bfloat16
        if cfg.train.bf16
        else torch.float16
        if cfg.train.fp16
        else torch.float32
    )
    optim_config = call_with_supported_kwargs(
        OptimizerConfig,
        optimizer="adam",
        lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
        weight_decay=cfg.train.weight_decay,
        adam_beta1=cfg.train.adam_beta1,
        adam_beta2=cfg.train.adam_beta2,
        adam_eps=cfg.train.adam_eps,
        clip_grad=cfg.train.clip_grad,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        params_dtype=params_dtype,
        use_distributed_optimizer=bool(cfg.train.use_distributed_optimizer),
    )
    optimizer_chunk = optimizer_model_chunk(model)
    optimizer = call_with_supported_kwargs(
        get_megatron_optimizer,
        config=optim_config,
        model_chunks=[optimizer_chunk],
        use_gloo_process_groups=False,
    )
    setattr(optimizer, "_dar_mcore_optimizer", True)
    return optimizer


def build_torch_optimizer(model: torch.nn.Module, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    return optimizer


def build_training_stack(base_model: torch.nn.Module, cfg, local_rank: int, dp_size: int):
    if prefer_mcore_training_stack(cfg):
        try:
            startup_log("build_training_stack wrap_megatron_ddp begin", all_ranks=True)
            model = wrap_megatron_ddp(base_model, cfg)
            startup_log("build_training_stack wrap_megatron_ddp done", all_ranks=True)
            startup_log("build_training_stack build_megatron_optimizer begin", all_ranks=True)
            optimizer = build_megatron_optimizer(model, cfg)
            startup_log("build_training_stack build_megatron_optimizer done", all_ranks=True)
            return model, optimizer, "mcore_ddp+mcore_optimizer"
        except Exception as exc:
            if not env_bool("MEGATRON_ALLOW_TORCH_FALLBACK", False):
                raise RuntimeError(
                    "MCore training stack failed. For paper-quality comparisons this is fatal; "
                    "set MEGATRON_USE_MCORE_TRAINING=0 to intentionally use the torch stack, "
                    "or MEGATRON_ALLOW_TORCH_FALLBACK=1 for debugging only."
                ) from exc
            if torch.distributed.get_rank() == 0:
                print(
                    "mcore training stack unavailable, falling back to torch path: "
                    f"{type(exc).__name__}: {exc}"
                )

    if dp_size > 1:
        model = wrap_ddp(base_model, local_rank)
        setattr(model, "_dar_training_stack", "torch_ddp")
    else:
        model = base_model
        setattr(model, "_dar_training_stack", "local")
    startup_log("build_training_stack build_torch_optimizer begin", all_ranks=True)
    optimizer = build_torch_optimizer(model, cfg)
    startup_log("build_training_stack build_torch_optimizer done", all_ranks=True)
    return model, optimizer, f"{getattr(model, '_dar_training_stack', 'unknown')}+torch_adamw"


def is_mcore_optimizer(optimizer) -> bool:
    return bool(getattr(optimizer, "_dar_mcore_optimizer", False))


def zero_training_gradients(model: torch.nn.Module, optimizer, reset_grad_buffers: bool = True) -> None:
    if reset_grad_buffers and hasattr(model, "zero_grad_buffer"):
        model.zero_grad_buffer()
    optimizer.zero_grad(set_to_none=True)


def finish_training_grad_sync(model: torch.nn.Module) -> None:
    if hasattr(model, "finish_grad_sync"):
        model.finish_grad_sync()
    finalize_sequence_parallel_grads(model)


def set_moe_aux_loss_scale(num_microbatches: int, context_parallel_size: int = 1) -> None:
    """Match Megatron pipeline schedule scaling for MoE aux-loss gradients."""
    if num_microbatches <= 0 or not torch.cuda.is_available():
        return
    try:
        from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
    except Exception:
        return
    scale = (
        torch.ones(1, device=torch.cuda.current_device())
        * max(1, int(context_parallel_size))
        / int(num_microbatches)
    )
    MoEAuxLossAutoScaler.set_loss_scale(scale)


def step_training_optimizer(optimizer):
    result = optimizer.step()
    if isinstance(result, tuple):
        update_successful = bool(result[0])
        grad_norm = result[1] if len(result) > 1 else None
        num_zeros = result[2] if len(result) > 2 else None
        return update_successful, grad_norm, num_zeros
    return True, None, None


def grad_debug_enabled() -> bool:
    return env_bool("MEGATRON_GRAD_DEBUG", False)


def _optimizer_lr_text(optimizer) -> str:
    value = _optimizer_lr_value(optimizer)
    return "na" if value is None else f"{value:.3e}"


def _optimizer_lr_value(optimizer) -> float | None:
    groups = getattr(optimizer, "param_groups", None)
    if groups:
        try:
            return float(groups[0].get("lr", float("nan")))
        except Exception:
            return None
    return None


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().float()
            if int(value.numel()) == 1:
                return float(value.item())
            return float(value.mean().item())
        return float(value)
    except Exception:
        return None


def wandb_enabled() -> bool:
    return env_bool("MEGATRON_ENABLE_WANDB", False)


def should_initialize_wandb() -> bool:
    if not wandb_enabled():
        return False
    if torch.distributed.is_initialized():
        return should_log_from_rank()
    return int(os.environ.get("RANK", "0")) == env_int("MEGATRON_WANDB_RANK", 0)


def wandb_init_if_enabled(
    cfg,
    config_path: str,
    training_stack: str = "pre_distributed",
    dp_size: int | None = None,
    pp_size: int | None = None,
    grad_accum_steps: int | None = None,
    lr_scheduler=None,
) -> None:
    global _WANDB_RUN
    if _WANDB_RUN is not None or not should_initialize_wandb():
        return
    try:
        import wandb
    except Exception as exc:
        message = f"[wandb] disabled import_failed={type(exc).__name__}: {exc}"
        print(message, flush=True)
        if env_bool("MEGATRON_WANDB_STRICT", False):
            raise
        return

    run_log_dir = os.environ.get("MEGATRON_RUN_LOG_DIR", "")
    default_name = Path(run_log_dir).name if run_log_dir else Path(config_path).stem
    project = os.environ.get("MEGATRON_WANDB_PROJECT") or os.environ.get("WANDB_PROJECT") or "dual-attentions-residuals"
    entity = os.environ.get("MEGATRON_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY") or None
    name = os.environ.get("MEGATRON_WANDB_NAME") or os.environ.get("WANDB_NAME") or default_name
    group = os.environ.get("MEGATRON_WANDB_GROUP") or os.environ.get("WANDB_GROUP") or None
    tags_text = os.environ.get("MEGATRON_WANDB_TAGS") or os.environ.get("WANDB_TAGS") or ""
    tags = [item.strip() for item in tags_text.split(",") if item.strip()]
    mode = os.environ.get("MEGATRON_WANDB_MODE") or os.environ.get("WANDB_MODE") or None
    wandb_dir = os.environ.get("MEGATRON_WANDB_DIR") or os.environ.get("WANDB_DIR") or run_log_dir or None
    run_id = os.environ.get("MEGATRON_WANDB_ID") or os.environ.get("WANDB_RUN_ID") or None
    resume = os.environ.get("MEGATRON_WANDB_RESUME") or os.environ.get("WANDB_RESUME") or None
    notes = os.environ.get("MEGATRON_WANDB_NOTES") or os.environ.get("WANDB_NOTES") or None
    if wandb_dir:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    init_timeout = env_float("MEGATRON_WANDB_INIT_TIMEOUT", 45.0)
    service_wait = env_float("MEGATRON_WANDB_SERVICE_WAIT", 20.0)
    login_timeout = env_float("MEGATRON_WANDB_LOGIN_TIMEOUT", 45.0)
    start_method = os.environ.get("MEGATRON_WANDB_START_METHOD", "").strip() or None
    disable_stats = env_bool("MEGATRON_WANDB_DISABLE_STATS", True)
    settings_kwargs = {
        "init_timeout": init_timeout,
        "login_timeout": login_timeout,
        "x_service_wait": service_wait,
        "x_disable_stats": disable_stats,
        "x_disable_machine_info": disable_stats,
        "disable_git": True,
        "disable_code": True,
        "console": "off",
    }
    if start_method is not None:
        settings_kwargs["start_method"] = start_method
    try:
        import inspect

        supported_settings = inspect.signature(wandb.Settings).parameters
        settings_kwargs = {key: value for key, value in settings_kwargs.items() if key in supported_settings}
        wandb_settings = wandb.Settings(**settings_kwargs)
    except Exception as exc:
        print(f"[wandb] settings_fallback reason={type(exc).__name__}: {exc}", flush=True)
        wandb_settings = None

    wandb_config = {
        "config_path": config_path,
        "model": dict(vars(cfg.model)),
        "parallel": dict(vars(cfg.parallel)),
        "train": dict(vars(cfg.train)),
        "runtime": {
            "training_stack": training_stack,
            "dp_size": dp_size,
            "pp_size": pp_size,
            "grad_accum_steps": grad_accum_steps,
            "world_size": (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "tokens_per_step": int(cfg.train.global_batch_size) * int(cfg.model.seq_length),
            "lr_scheduler_enabled": lr_scheduler is not None,
            "run_log_dir": run_log_dir,
        },
    }
    init_kwargs = {
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "tags": tags or None,
        "mode": mode,
        "dir": wandb_dir,
        "id": run_id,
        "resume": resume,
        "notes": notes,
        "config": wandb_config,
    }
    if wandb_settings is not None:
        init_kwargs["settings"] = wandb_settings
    init_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, "", [])}
    print(
        f"[wandb] init begin project={project} name={name} mode={mode or 'default'} "
        f"init_timeout={init_timeout:.1f}s service_wait={service_wait:.1f}s",
        flush=True,
    )
    try:
        _WANDB_RUN = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[wandb] disabled init_failed={type(exc).__name__}: {exc}", flush=True)
        if env_bool("MEGATRON_WANDB_STRICT", False):
            raise
        _WANDB_RUN = None
        return
    try:
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("perf/*", step_metric="iter")
    except Exception:
        pass
    atexit.register(wandb_finish)
    print(f"[wandb] enabled project={project} name={name} dir={wandb_dir or 'default'}", flush=True)


def wandb_barrier_if_enabled() -> None:
    if wandb_enabled() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def wandb_log(step: int, metrics: dict[str, object]) -> None:
    if _WANDB_RUN is None:
        return
    clean: dict[str, float | int] = {"iter": int(step)}
    for key, value in metrics.items():
        scalar = _float_or_none(value)
        if scalar is not None and math.isfinite(scalar):
            clean[key] = scalar
    if len(clean) > 1:
        _WANDB_RUN.log(clean, step=int(step))


def wandb_finish() -> None:
    global _WANDB_RUN
    if _WANDB_RUN is None:
        return
    try:
        _WANDB_RUN.finish()
    except Exception:
        pass
    _WANDB_RUN = None


def lr_scheduler_enabled() -> bool:
    return env_bool("MEGATRON_ENABLE_LR_SCHEDULER", False)


def lr_scheduler_settings(cfg) -> tuple[bool, int, int, str, float]:
    enabled = lr_scheduler_enabled()
    warmup_iters = max(0, env_int("MEGATRON_LR_WARMUP_ITERS", 0))
    decay_iters = env_int("MEGATRON_LR_DECAY_ITERS", int(cfg.train.train_iters))
    decay_iters = max(warmup_iters + 1, decay_iters)
    style = os.environ.get("MEGATRON_LR_DECAY_STYLE", "cosine").strip().lower()
    warmup_init_lr = env_float("MEGATRON_LR_WARMUP_INIT_LR", 0.0)
    return enabled, warmup_iters, decay_iters, style, warmup_init_lr


def build_optimizer_lr_scheduler(optimizer, cfg):
    enabled, warmup_iters, decay_iters, style, warmup_init_lr = lr_scheduler_settings(cfg)
    if not enabled:
        return None
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    return OptimizerParamScheduler(
        optimizer,
        init_lr=warmup_init_lr,
        max_lr=float(cfg.train.lr),
        min_lr=float(cfg.train.min_lr),
        lr_warmup_steps=warmup_iters,
        lr_decay_steps=decay_iters,
        lr_decay_style=style,
        start_wd=float(cfg.train.weight_decay),
        end_wd=float(cfg.train.weight_decay),
        wd_incr_steps=max(1, int(cfg.train.train_iters)),
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )


def step_optimizer_lr_scheduler(lr_scheduler, optimizer, update_successful: bool) -> str:
    if lr_scheduler is not None and update_successful:
        lr_scheduler.step(1)
    return _optimizer_lr_text(optimizer)


def _param_grad_tensor(param: torch.nn.Parameter):
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


_SP_REPLICATED_GRAD_SYNC_PROFILE_STEP = 0
_SP_REPLICATED_GRAD_SYNC_DISABLE_WARNED = False


def finalize_sequence_parallel_grads(model: torch.nn.Module) -> None:
    global _SP_REPLICATED_GRAD_SYNC_PROFILE_STEP
    global _SP_REPLICATED_GRAD_SYNC_DISABLE_WARNED
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
    except Exception:
        return
    if tp_group is None or tp_size <= 1:
        return
    if env_bool("MEGATRON_UNSAFE_DISABLE_REPLICATED_TP_GRAD_SYNC", False):
        if not _SP_REPLICATED_GRAD_SYNC_DISABLE_WARNED and torch.distributed.get_rank() == 0:
            print(
                "[replicated_tp_grad_sync] WARNING disabled by "
                "MEGATRON_UNSAFE_DISABLE_REPLICATED_TP_GRAD_SYNC=1; use only for speed diagnosis",
                flush=True,
            )
            _SP_REPLICATED_GRAD_SYNC_DISABLE_WARNED = True
        return
    profile = env_bool("MEGATRON_REPLICATED_TP_GRAD_SYNC_PROFILE", False)
    profile_steps = env_int("MEGATRON_REPLICATED_TP_GRAD_SYNC_PROFILE_STEPS", 5)
    if profile:
        _SP_REPLICATED_GRAD_SYNC_PROFILE_STEP += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sync_start = time.perf_counter()
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    param_count = 0
    grad_count = 0
    grad_numel = 0
    for param in model.parameters():
        if not bool(getattr(param, "sequence_parallel", False)):
            continue
        param_count += 1
        grad = _param_grad_tensor(param)
        if grad is not None:
            grad_count += 1
            grad_numel += int(grad.numel())
            buckets.setdefault((grad.device, grad.dtype), []).append(grad)
    for grads in buckets.values():
        if len(grads) == 1:
            torch.distributed.all_reduce(grads[0], op=torch.distributed.ReduceOp.SUM, group=tp_group)
            continue
        flat = torch.cat([grad.detach().contiguous().view(-1) for grad in grads])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(flat[offset : offset + numel].view_as(grad))
            offset += numel
    if profile:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - sync_start
        if _SP_REPLICATED_GRAD_SYNC_PROFILE_STEP <= profile_steps and torch.distributed.get_rank() == 0:
            print(
                "[replicated_tp_grad_sync] "
                f"step={_SP_REPLICATED_GRAD_SYNC_PROFILE_STEP} "
                f"params={param_count} grads={grad_count} numel={grad_numel} "
                f"buckets={len(buckets)} elapsed={elapsed:.6f}s",
                flush=True,
            )


def _grad_stats_for_names(model: torch.nn.Module, patterns: tuple[str, ...]) -> tuple[int, float, float, int]:
    count = 0
    main_grad_count = 0
    sq_sum = 0.0
    max_abs = 0.0
    for name, param in model.named_parameters():
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        grad = _param_grad_tensor(param)
        if grad is None:
            continue
        if getattr(param, "main_grad", None) is not None:
            main_grad_count += 1
        grad_f = grad.detach().float()
        count += 1
        sq_sum += float(grad_f.pow(2).sum().item())
        max_abs = max(max_abs, float(grad_f.abs().max().item()))
    return count, math.sqrt(sq_sum), max_abs, main_grad_count


def print_grad_debug(model: torch.nn.Module, optimizer, step: int) -> None:
    if not grad_debug_enabled() or step > int(os.environ.get("MEGATRON_GRAD_DEBUG_STEPS", "5")):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    groups = {
        "all": (),
        "embed": ("embedding", "word_embeddings"),
        "output": ("output_layer", "lm_head"),
        "attres_proj": (
            "attn_res_proj",
            "mlp_res_proj",
            "attn_res_k_proj",
            "mlp_res_k_proj",
            "output_aggregation.proj",
            "final_block_output_aggregation.proj",
        ),
        "dual_mix": (
            "attn_q",
            "mlp_q",
            "static_alpha",
            "static_beta",
            "dynamic_ab_fn",
            "dynamic_residual_mix_fn",
            "static_residual_logit",
            "pre_branch_scale",
            "h_post_scale",
            "residual_scale",
        ),
        "transformer": ("self_attention", ".attention", ".mlp", "mlp.", "linear_fc"),
    }
    parts = [f"[grad_debug rank={rank}] step={step} lr={_optimizer_lr_text(optimizer)}"]
    for label, patterns in groups.items():
        count, norm, max_abs, main_grad_count = _grad_stats_for_names(model, patterns)
        parts.append(f"{label}:n={count},main={main_grad_count},norm={norm:.3e},max={max_abs:.3e}")
    print(" ".join(parts), flush=True)


def print_optimizer_debug(step: int, update_successful: bool, grad_norm, num_zeros) -> None:
    if not grad_debug_enabled() or step > int(os.environ.get("MEGATRON_GRAD_DEBUG_STEPS", "5")):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _fmt(value) -> str:
        if value is None:
            return "none"
        if hasattr(value, "detach"):
            try:
                return f"{float(value.detach().float().item()):.3e}"
            except Exception:
                return str(value)
        try:
            return f"{float(value):.3e}"
        except Exception:
            return str(value)

    print(
        f"[optim_debug rank={rank}] step={step} update_successful={update_successful} "
        f"grad_norm={_fmt(grad_norm)} num_zeros={_fmt(num_zeros)}",
        flush=True,
    )


def capture_update_debug(model: torch.nn.Module):
    if not grad_debug_enabled():
        return None
    patterns = (
        "output_layer",
        "word_embeddings",
        "attn_res_proj",
        "mlp_res_proj",
        "attn_q",
        "mlp_q",
        "dynamic_ab_fn",
        "linear_fc",
    )
    samples = []
    for name, param in model.named_parameters():
        if not any(pattern in name for pattern in patterns):
            continue
        data = param.detach().float().reshape(-1)
        if data.numel() == 0:
            continue
        samples.append((name, param, data[: min(256, data.numel())].clone()))
        if len(samples) >= 6:
            break
    return samples


def print_update_debug(step: int, before_samples) -> None:
    if before_samples is None or not grad_debug_enabled() or step > int(os.environ.get("MEGATRON_GRAD_DEBUG_STEPS", "5")):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    parts = [f"[update_debug rank={rank}] step={step}"]
    for name, param, before in before_samples:
        after = param.detach().float().reshape(-1)[: before.numel()]
        delta = (after - before).abs()
        parts.append(f"{name}:mean={float(delta.mean().item()):.3e},max={float(delta.max().item()):.3e}")
    print(" ".join(parts), flush=True)


def dual_tp_debug_enabled() -> bool:
    return env_bool("MEGATRON_DUAL_TP_DEBUG", False)


def _dual_tp_debug_limit() -> int:
    return env_int("MEGATRON_DUAL_TP_DEBUG_STEPS", env_int("MEGATRON_GRAD_DEBUG_STEPS", 5))


def _dual_tp_debug_patterns() -> tuple[str, ...]:
    return (
        "attn_res_proj",
        "mlp_res_proj",
        "attn_res_k_proj",
        "mlp_res_k_proj",
        "attn_q",
        "mlp_q",
        "attn_dynamic_ab_fn",
        "mlp_dynamic_ab_fn",
        "attn_dynamic_residual_mix_fn",
        "mlp_dynamic_residual_mix_fn",
        "attn_static_alpha",
        "mlp_static_alpha",
        "attn_static_beta",
        "mlp_static_beta",
        "attn_static_residual_logit",
        "mlp_static_residual_logit",
        "pre_branch_scale",
        "h_post_scale",
        "residual_scale",
        "output_aggregation.q",
        "output_aggregation.proj",
        "final_block_output_aggregation.proj",
    )


def _tp_debug_info():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 1, None, 0
    try:
        from megatron.core import parallel_state

        tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
        tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_src_rank = int(parallel_state.get_tensor_model_parallel_src_rank())
        return tp_rank, tp_size, tp_group, tp_src_rank
    except Exception:
        return 0, 1, None, int(torch.distributed.get_rank())


def _tp_group_max_diff(tensor: torch.Tensor, *, group, src_rank: int) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized() or group is None:
        return 0.0
    ref = tensor.detach().float().contiguous().clone()
    torch.distributed.broadcast(ref, src=src_rank, group=group)
    local = (tensor.detach().float() - ref).abs().max()
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX, group=group)
    return float(local.item())


def sync_sequence_parallel_parameters(model: torch.nn.Module) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_src_rank = int(parallel_state.get_tensor_model_parallel_src_rank())
        tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
    except Exception:
        return
    if tp_group is None or tp_size <= 1:
        return
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            if bool(getattr(param, "sequence_parallel", False)):
                torch.distributed.broadcast(param.data, src=tp_src_rank, group=tp_group)
                count += 1
    if count > 0 and torch.distributed.get_rank() == tp_src_rank:
        print(f"[dual_tp_sync] broadcast_sequence_parallel_params={count}", flush=True)


def print_dual_tp_debug(model: torch.nn.Module, step: int, phase: str, *, include_grads: bool) -> None:
    if not dual_tp_debug_enabled() or step > _dual_tp_debug_limit():
        return
    tp_rank, tp_size, tp_group, tp_src_rank = _tp_debug_info()
    if tp_size <= 1:
        return
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    patterns = _dual_tp_debug_patterns()
    param_count = 0
    grad_count = 0
    max_param_diff = 0.0
    max_grad_diff = 0.0
    worst_param = "none"
    worst_grad = "none"
    missing_sequence_parallel: list[str] = []
    for name, param in model.named_parameters():
        if not any(pattern in name for pattern in patterns):
            continue
        param_count += 1
        if not bool(getattr(param, "sequence_parallel", False)):
            missing_sequence_parallel.append(name)
        param_diff = _tp_group_max_diff(param.data, group=tp_group, src_rank=tp_src_rank)
        if param_diff > max_param_diff:
            max_param_diff = param_diff
            worst_param = name
        if include_grads:
            grad = _param_grad_tensor(param)
            if grad is not None:
                grad_count += 1
                grad_diff = _tp_group_max_diff(grad, group=tp_group, src_rank=tp_src_rank)
                if grad_diff > max_grad_diff:
                    max_grad_diff = grad_diff
                    worst_grad = name
    missing_preview = ",".join(missing_sequence_parallel[:4])
    if len(missing_sequence_parallel) > 4:
        missing_preview += f",+{len(missing_sequence_parallel) - 4}"
    if tp_rank == 0:
        print(
            f"[dual_tp_debug rank={global_rank} tp_rank={tp_rank}/{tp_size}] "
            f"step={step} phase={phase} params={param_count} grads={grad_count} "
            f"param_max_diff={max_param_diff:.3e} worst_param={worst_param} "
            f"grad_max_diff={max_grad_diff:.3e} worst_grad={worst_grad} "
            f"missing_sequence_parallel={len(missing_sequence_parallel)}:{missing_preview}",
            flush=True,
        )


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        return tensor
    out = tensor.detach().clone()
    group = data_parallel_group()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=group)
    out /= max(1, data_parallel_world_size())
    return out


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(max(0, num_bytes))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}GiB"


def progress_bar(step: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    ratio = min(1.0, max(0.0, float(step) / float(total)))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {ratio * 100.0:5.1f}%"


def read_and_reset_moe_load_balancing_loss(normalizer: int = 1) -> torch.Tensor | None:
    """Read Megatron-Core MoE load-balancing metrics without affecting gradients."""
    if not torch.distributed.is_initialized():
        return None
    try:
        from megatron.core import parallel_state
        from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker

        tracker = get_moe_layer_wise_logging_tracker()
    except Exception:
        return None
    if not tracker:
        return None

    def _entry_attr(entry, name: str):
        if isinstance(entry, dict):
            return entry.get(name)
        return getattr(entry, name, None)

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)
    count = torch.zeros((), device=device, dtype=torch.float32)
    for name, entry in tracker.items():
        if "load_balancing_loss" not in str(name):
            continue
        values = _entry_attr(entry, "values")
        if values is None:
            continue
        values = values.detach().float().clone()
        try:
            torch.distributed.all_reduce(values, group=parallel_state.get_pipeline_model_parallel_group())
        except Exception:
            pass
        reduce_group = _entry_attr(entry, "reduce_group")
        if reduce_group is not None:
            try:
                torch.distributed.all_reduce(values, group=reduce_group)
            except Exception:
                pass
        avg_group = _entry_attr(entry, "avg_group")
        if avg_group is not None:
            try:
                torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.AVG, group=avg_group)
            except Exception:
                pass
        values = values / max(1, int(normalizer))
        mask = values != 0
        if bool(mask.any()):
            total = total + values[mask].sum()
            count = count + mask.sum().to(dtype=torch.float32, device=device)

    for entry in tracker.values():
        values = _entry_attr(entry, "values")
        if values is not None:
            values.zero_()

    if float(count.item()) == 0.0:
        return None
    return total / count


def clip_grad_norm_global_(parameters, max_norm: float, dp_size: int = 1, eps: float = 1.0e-6) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    device = (
        params[0].grad.device
        if params
        else torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    local_sq = torch.zeros((), device=device, dtype=torch.float32)
    for param in params:
        grad = param.grad.detach().float()
        local_sq = local_sq + grad.pow(2).sum()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(local_sq, op=torch.distributed.ReduceOp.SUM)
        local_sq = local_sq / max(1, int(dp_size))
    total_norm = local_sq.sqrt()
    clip_coef = float(max_norm) / (total_norm + eps)
    if clip_coef < 1.0:
        for param in params:
            param.grad.detach().mul_(clip_coef)
    return total_norm


def attnres_package_shape(cfg, micro_batch_size: int) -> tuple[int, int, int, int]:
    block_layers = max(1, int(cfg.model.hyper_conn_block_layers))
    num_blocks = cfg.model.num_layers // block_layers
    streams = 2 if cfg.model.variant.lower() == "dual_attention_residual_block" else 1
    seq_length = cfg.model.seq_length
    if bool(cfg.parallel.sequence_parallel) and int(cfg.parallel.tensor_model_parallel_size) > 1:
        seq_length = seq_length // int(cfg.parallel.tensor_model_parallel_size)
    return (num_blocks + 1, seq_length, micro_batch_size * streams, cfg.model.hidden_size)


def attnres_stage_package_shapes(cfg, micro_batch_size: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    block_layers = max(1, int(cfg.model.hyper_conn_block_layers))
    num_blocks = cfg.model.num_layers // block_layers
    pp_size = pipeline_world_size()
    try:
        from megatron.core import parallel_state

        pp_rank = int(parallel_state.get_pipeline_model_parallel_rank())
    except Exception:
        pp_rank = 0
    if num_blocks % pp_size != 0:
        raise ValueError(f"num_blocks={num_blocks} must be divisible by pipeline_world_size={pp_size}")
    blocks_per_stage = num_blocks // pp_size
    stage_start_block = pp_rank * blocks_per_stage
    stage_end_block = (pp_rank + 1) * blocks_per_stage
    streams = 2 if cfg.model.variant.lower() == "dual_attention_residual_block" else 1
    seq_length = cfg.model.seq_length
    if bool(cfg.parallel.sequence_parallel) and int(cfg.parallel.tensor_model_parallel_size) > 1:
        seq_length = seq_length // int(cfg.parallel.tensor_model_parallel_size)

    def shape(slots: int) -> tuple[int, int, int, int]:
        return (slots, seq_length, micro_batch_size * streams, cfg.model.hidden_size)

    return shape(stage_start_block + 1), shape(stage_end_block + 1)


def tensor_shape_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * element_size


def estimated_pp_bubble_fraction(pp_size: int, num_microbatches: int) -> float:
    if pp_size <= 1 or num_microbatches <= 0:
        return 0.0
    return float(pp_size - 1) / float(num_microbatches + pp_size - 1)


def make_adjust_tensor_shapes_fn(cfg):
    if cfg.model.variant.lower() not in {"attention_residual_block", "dual_attention_residual_block"}:
        return None
    recv_package_shape, send_package_shape = attnres_stage_package_shapes(cfg, cfg.train.micro_batch_size)

    def _replace(shapes, package_shape):
        if shapes is None:
            return None
        if isinstance(shapes, list):
            return [package_shape if shape is not None else None for shape in shapes]
        return package_shape

    def adjust_tensor_shapes_fn(recv_tensor_shapes, send_tensor_shapes):
        return _replace(recv_tensor_shapes, recv_package_shape), _replace(send_tensor_shapes, send_package_shape)

    return adjust_tensor_shapes_fn


_CAUSAL_MASK_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def make_left_to_right_attention_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Match Megatron GPTDataset's bool causal mask: True entries are masked."""
    seq_length = int(tokens.shape[1])
    key = (tokens.device, seq_length)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        visible = torch.tril(torch.ones((seq_length, seq_length), device=tokens.device, dtype=torch.bool))
        mask = (~visible).unsqueeze(0).unsqueeze(0)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def make_position_ids(tokens: torch.Tensor) -> torch.Tensor:
    return torch.arange(tokens.shape[1], dtype=torch.long, device=tokens.device).unsqueeze(0).expand_as(tokens)


def forward_step_func(data_iterator, model):
    if isinstance(data_iterator, list):
        data_iterator = data_iterator[0]
    batch = next(data_iterator)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    batch = batch.to(device, non_blocking=True)
    tokens, labels = batch_to_tokens_and_labels(batch)
    attention_mask = make_left_to_right_attention_mask(tokens)
    position_ids = make_position_ids(tokens)
    with autocast_context_for_model(model):
        output = model(tokens, labels=labels, attention_mask=attention_mask, position_ids=position_ids)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)

    def loss_func(output_tensor):
        if not isinstance(output_tensor, dict):
            losses = output_tensor.float()
            token_losses = losses.view(-1)
            flat_mask = loss_mask.view(-1).float()
            loss_sum = torch.sum(token_losses * flat_mask)
            num_tokens = flat_mask.sum().detach().to(torch.int)
            mean_loss = loss_sum.detach() / num_tokens.clamp_min(1).to(dtype=loss_sum.dtype)
            report = {
                "loss": mean_loss.view(1),
                "lm_loss": mean_loss.view(1),
                "lm loss": torch.cat([loss_sum.detach().view(1), num_tokens.view(1)]),
            }
            return loss_sum, num_tokens, report
        loss = output_tensor["loss"]
        metrics = {
            "loss": loss.detach(),
            "lm_loss": output_tensor["lm_loss"].detach(),
            "aux_loss": output_tensor["aux_loss"].detach(),
        }
        return loss, metrics

    return output, loss_func


def mean_loss_from_model_output(output, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(output, dict):
        return output["loss"]
    losses = output.float()
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    token_losses = losses.view(-1)
    flat_mask = loss_mask.view(-1).float()
    return torch.sum(token_losses * flat_mask) / flat_mask.sum().clamp_min(1).to(dtype=losses.dtype)


def extract_scheduler_metric(losses_reduced, key: str = "loss") -> torch.Tensor | None:
    if not losses_reduced:
        return None
    for item in reversed(losses_reduced):
        if isinstance(item, dict):
            value = item.get(key)
            if value is None and key == "lm_loss":
                value = item.get("lm loss")
            if value is None and key == "loss":
                value = item.get("lm_loss")
                if value is None:
                    value = item.get("lm loss")
            if value is not None:
                if hasattr(value, "numel") and int(value.numel()) == 2:
                    value = value[0] / value[1].clamp_min(1).to(dtype=value.dtype)
                return value
        if key == "loss" and hasattr(item, "shape"):
            return item
    return None


def checkpoint_saving_enabled() -> bool:
    return not env_bool("MEGATRON_DISABLE_SAVE", False)


def maybe_save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    step: int,
) -> None:
    if not checkpoint_saving_enabled():
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    ckpt_dir = Path(path) / f"iter_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    overlap_ddp = getattr(model, "ddp", None)
    restore_forward_pre_hook = bool(
        overlap_ddp is not None
        and getattr(overlap_ddp, "use_forward_hook", False)
        and hasattr(overlap_ddp, "disable_forward_pre_hook")
        and hasattr(overlap_ddp, "enable_forward_pre_hook")
    )
    if restore_forward_pre_hook:
        overlap_ddp.disable_forward_pre_hook(param_sync=True)
    try:
        module = unwrap_model(model)
        payload = {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(payload, ckpt_dir / f"rank_{rank:05d}.pt")
        if hasattr(optimizer, "save_parameter_state"):
            optimizer.save_parameter_state(
                str(ckpt_dir / f"optimizer_parameter_state_rank_{rank:05d}.pt")
            )
    finally:
        if restore_forward_pre_hook:
            overlap_ddp.enable_forward_pre_hook()
    if rank == 0:
        latest_tmp = Path(path) / "latest.tmp"
        latest_tmp.write_text(ckpt_dir.name + "\n", encoding="utf-8")
        latest_tmp.replace(Path(path) / "latest")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        print(f"[checkpoint] saved step={step} dir={ckpt_dir}", flush=True)


def resolve_checkpoint_file(load_path: str) -> Path:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(load_path)
    if path.is_file():
        return path
    if (path / "latest").is_file():
        latest = (path / "latest").read_text(encoding="utf-8").strip()
        if latest:
            path = path / latest
    rank_file = path / f"rank_{rank:05d}.pt"
    if rank_file.is_file():
        return rank_file
    if torch.distributed.is_initialized():
        raise FileNotFoundError(f"Missing checkpoint file {rank_file} for rank={rank}")
    candidates = sorted(path.glob("rank_*.pt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No rank checkpoint file found under {load_path!r} for rank={rank}")


def cuda_memory_text() -> str:
    if not torch.cuda.is_available():
        return "cuda=na"
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    free = total = 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free = free_bytes / (1024**3)
        total = total_bytes / (1024**3)
    except Exception:
        pass
    return f"cuda_alloc={allocated:.2f}GiB cuda_reserved={reserved:.2f}GiB cuda_free={free:.2f}/{total:.2f}GiB"


def cleanup_after_checkpoint_load() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_checkpoint(
    load_path: str,
    model: torch.nn.Module,
    optimizer,
    lr_scheduler,
) -> int:
    ckpt_file = resolve_checkpoint_file(load_path)
    startup_log(f"checkpoint load begin file={ckpt_file} {cuda_memory_text()}", all_ranks=True)
    try:
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    saved_world_size = checkpoint.get("world_size")
    if saved_world_size is not None and int(saved_world_size) != world_size:
        raise RuntimeError(
            f"Checkpoint world_size={saved_world_size} does not match current world_size={world_size}"
        )
    saved_rank = checkpoint.get("rank")
    if saved_rank is not None and int(saved_rank) != rank:
        raise RuntimeError(f"Checkpoint rank={saved_rank} does not match current rank={rank}: {ckpt_file}")
    module = unwrap_model(model)
    missing, unexpected = module.load_state_dict(checkpoint["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint model keys mismatch: missing={missing} unexpected={unexpected}")
    startup_log(f"checkpoint model loaded {cuda_memory_text()}", all_ranks=True)

    load_optimizer = env_bool("MEGATRON_LOAD_OPTIMIZER", True)
    if load_optimizer and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if hasattr(optimizer, "load_parameter_state"):
            optimizer.load_parameter_state(
                str(ckpt_file.parent / f"optimizer_parameter_state_rank_{rank:05d}.pt")
            )
        startup_log(f"checkpoint optimizer loaded {cuda_memory_text()}", all_ranks=True)
    elif not load_optimizer:
        startup_log("checkpoint optimizer load skipped by MEGATRON_LOAD_OPTIMIZER=0", all_ranks=True)

    if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    step = int(checkpoint.get("step", 0))
    del checkpoint
    cleanup_after_checkpoint_load()
    startup_log(f"checkpoint loaded file={ckpt_file} step={step} {cuda_memory_text()}", all_ranks=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return step


def evaluation_enabled(cfg) -> bool:
    if env_bool("MEGATRON_DISABLE_EVAL", False):
        return False
    return int(getattr(cfg.train, "eval_interval", 0) or 0) > 0 and int(getattr(cfg.train, "eval_iters", 0) or 0) > 0


def evaluate_no_pipeline(
    model: torch.nn.Module,
    iterator,
    cfg,
    device: torch.device,
    grad_accum_steps: int,
) -> tuple[float | None, float | None]:
    total_loss = 0.0
    valid_losses = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(int(cfg.train.eval_iters)):
                step_loss = None
                for _micro_idx in range(grad_accum_steps):
                    batch = next(iterator).to(device, non_blocking=True)
                    tokens, labels = batch_to_tokens_and_labels(batch)
                    attention_mask = make_left_to_right_attention_mask(tokens)
                    position_ids = make_position_ids(tokens)
                    with autocast_context_for_model(model):
                        out = model(tokens, labels=labels, attention_mask=attention_mask, position_ids=position_ids)
                    loss = mean_loss_from_model_output(out, labels) / grad_accum_steps
                    loss = loss.detach()
                    step_loss = loss if step_loss is None else step_loss + loss
                mean_loss = reduce_mean(step_loss)
                total_loss += float(mean_loss)
                valid_losses += 1
    finally:
        if was_training:
            model.train()

    tracker_lb_loss = read_and_reset_moe_load_balancing_loss(
        normalizer=max(1, int(cfg.train.eval_iters) * grad_accum_steps)
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if valid_losses == 0:
        return None, None if tracker_lb_loss is None else float(tracker_lb_loss)
    return total_loss / valid_losses, None if tracker_lb_loss is None else float(tracker_lb_loss)


def evaluate_pipeline(
    model: torch.nn.Module,
    iterator,
    cfg,
    grad_accum_steps: int,
    forward_backward_func,
    adjust_tensor_shapes_fn,
) -> tuple[float | None, float | None]:
    total_loss = 0.0
    valid_losses = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(int(cfg.train.eval_iters)):
                losses_reduced = call_with_supported_kwargs(
                    forward_backward_func,
                    forward_step_func=forward_step_func,
                    data_iterator=[iterator],
                    model=[model],
                    num_microbatches=grad_accum_steps,
                    seq_length=cfg.model.seq_length,
                    micro_batch_size=cfg.train.micro_batch_size,
                    decoder_seq_length=cfg.model.seq_length,
                    forward_only=True,
                    collect_non_loss_data=False,
                    adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
                )
                loss = extract_scheduler_metric(losses_reduced, "loss")
                if loss is not None:
                    total_loss += float(loss)
                    valid_losses += 1
    finally:
        if was_training:
            model.train()

    tracker_lb_loss = read_and_reset_moe_load_balancing_loss(
        normalizer=max(1, int(cfg.train.eval_iters) * grad_accum_steps)
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if valid_losses == 0:
        return None, None if tracker_lb_loss is None else float(tracker_lb_loss)
    return total_loss / valid_losses, None if tracker_lb_loss is None else float(tracker_lb_loss)


def print_validation_result(step: int, cfg, val_loss: float | None, val_lb_loss: float | None, elapsed: float) -> None:
    if not should_log_from_rank():
        return
    loss_text = "nan" if val_loss is None else f"{float(val_loss):.6f}"
    lb_text = "nan" if val_lb_loss is None else f"{float(val_lb_loss):.6f}"
    print(
        f"[validation] iter={step}/{cfg.train.train_iters} "
        f"val_loss={loss_text} val_lb_loss={lb_text} eval_iters={int(cfg.train.eval_iters)} "
        f"elapsed={format_duration(elapsed)}",
        flush=True,
    )
    wandb_log(
        step,
        {
            "val/loss": val_loss,
            "val/lb_loss": val_lb_loss,
            "val/eval_iters": int(cfg.train.eval_iters),
            "val/elapsed_seconds": elapsed,
        },
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = model
    while hasattr(module, "module"):
        module = module.module
    return module


def model_train_config(model: torch.nn.Module):
    if isinstance(model, list):
        model = model[0]
    module = unwrap_model(model)
    return getattr(module, "cfg", None)


def autocast_context_for_model(model: torch.nn.Module):
    cfg = model_train_config(model)
    if cfg is None or not torch.cuda.is_available():
        return nullcontext()
    if bool(getattr(cfg.train, "bf16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if bool(getattr(cfg.train, "fp16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def startup_log(message: str, *, all_ranks: bool = False) -> None:
    if os.environ.get("MEGATRON_STARTUP_DEBUG", "1").lower() in {"0", "false", "no"}:
        return
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if not all_ranks and rank != 0:
            return
        prefix = f"[startup rank={rank}]"
    else:
        prefix = f"[startup local_rank={os.environ.get('LOCAL_RANK', '?')}]"
    print(f"{prefix} {message}", flush=True)


def profile_enabled() -> bool:
    return os.environ.get("MEGATRON_STAGE_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def dual_block_profile_enabled() -> bool:
    return os.environ.get("MEGATRON_DUAL_BLOCK_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def attres_block_profile_enabled() -> bool:
    return os.environ.get("MEGATRON_ATTRES_BLOCK_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def baseline_layer_profile_enabled() -> bool:
    return os.environ.get("MEGATRON_BASELINE_LAYER_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def should_print_dual_profile() -> bool:
    if not torch.distributed.is_initialized():
        return True
    if os.environ.get("MEGATRON_DUAL_PROFILE_ALL_RANKS", "0").lower() in {"1", "true", "yes", "on"}:
        return True
    return should_log_from_rank()


def format_block_profile_detail(stats: dict[str, tuple[float, int, float]]) -> str:
    total_time = sum(total for total, _count, _avg in stats.values())
    denom = max(total_time, 1.0e-12)
    return " ".join(
        f"{name}={total:.3f}s/{count}({avg * 1000.0:.1f}ms,{total / denom * 100.0:.1f}%)"
        for name, (total, count, avg) in sorted(stats.items())
    )


def print_residual_block_profiles(step: int) -> None:
    if dual_block_profile_enabled() and should_print_dual_profile():
        try:
            from dual_attentions_residuals.residual_blocks import pop_dual_profile_stats

            dual_stats = pop_dual_profile_stats()
        except Exception:
            dual_stats = {}
        if dual_stats:
            rank_text = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            detail = format_block_profile_detail(dual_stats)
            print(f"[dual_block_profile rank={rank_text}] iter={step} {detail}", flush=True)
    if attres_block_profile_enabled() and should_print_dual_profile():
        try:
            from dual_attentions_residuals.residual_blocks import pop_attres_profile_stats

            attres_stats = pop_attres_profile_stats()
        except Exception:
            attres_stats = {}
        if attres_stats:
            rank_text = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            detail = format_block_profile_detail(attres_stats)
            print(f"[attres_block_profile rank={rank_text}] iter={step} {detail}", flush=True)


_BASELINE_LAYER_PROFILE_STATS: dict[str, list[float]] = {}


def _record_baseline_layer_profile(name: str, elapsed: float) -> None:
    values = _BASELINE_LAYER_PROFILE_STATS.setdefault(name, [0.0, 0.0])
    values[0] += float(elapsed)
    values[1] += 1.0


def pop_baseline_layer_profile_stats() -> dict[str, tuple[float, int, float]]:
    stats = {
        name: (values[0], int(values[1]), values[0] / max(1.0, values[1]))
        for name, values in _BASELINE_LAYER_PROFILE_STATS.items()
    }
    _BASELINE_LAYER_PROFILE_STATS.clear()
    return stats


def attach_baseline_layer_profile_hooks(model: torch.nn.Module, cfg) -> int:
    if not baseline_layer_profile_enabled() or cfg.model.variant.lower() != "baseline":
        return 0

    handles = []

    def make_pre_hook(label: str):
        def hook(module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            module._dar_profile_start = time.perf_counter()

        return hook

    def make_post_hook(label: str):
        def hook(module, _inputs, _output):
            start_time = getattr(module, "_dar_profile_start", None)
            if start_time is None:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _record_baseline_layer_profile(label, time.perf_counter() - start_time)

        return hook

    def attach(module: torch.nn.Module, label: str) -> None:
        handles.append(module.register_forward_pre_hook(make_pre_hook(label)))
        handles.append(module.register_forward_hook(make_post_hook(label)))

    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name == "TransformerLayer":
            attach(module, "layer_total")
            attention = getattr(module, "self_attention", None)
            mlp = getattr(module, "mlp", None)
            if isinstance(attention, torch.nn.Module):
                attach(attention, "attention")
            if isinstance(mlp, torch.nn.Module):
                attach(mlp, "mlp")

    # Keep hook handles alive with the model. This is diagnostic-only and enabled
    # explicitly by MEGATRON_BASELINE_LAYER_PROFILE.
    setattr(model, "_dar_baseline_layer_profile_handles", handles)
    return len(handles) // 2


def print_baseline_layer_profile(step: int) -> None:
    if not baseline_layer_profile_enabled() or not should_log_from_rank():
        return
    stats = pop_baseline_layer_profile_stats()
    if not stats:
        return
    total_layer = stats.get("layer_total", (0.0, 0, 0.0))[0]
    attention = stats.get("attention", (0.0, 0, 0.0))[0]
    mlp = stats.get("mlp", (0.0, 0, 0.0))[0]
    other = max(0.0, total_layer - attention - mlp)
    if total_layer > 0.0:
        stats = {
            **stats,
            "other_in_layer": (
                other,
                stats.get("layer_total", (0.0, 0, 0.0))[1],
                other / max(1, stats.get("layer_total", (0.0, 0, 0.0))[1]),
            ),
        }
    rank_text = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    detail = format_block_profile_detail(stats)
    print(f"[baseline_layer_profile rank={rank_text}] iter={step} {detail}", flush=True)


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer_now() -> float:
    sync_device()
    return time.perf_counter()


def main() -> None:
    args = parse_args()
    startup_log(f"load_config begin config={args.config}", all_ranks=True)
    cfg = load_config(args.config)
    if args.train_iters is not None:
        cfg.train.train_iters = args.train_iters
    if args.log_interval is not None:
        cfg.train.log_interval = args.log_interval
    if args.save_interval is not None:
        cfg.train.save_interval = args.save_interval
    if args.eval_interval is not None:
        cfg.train.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        cfg.train.eval_iters = args.eval_iters
    if args.micro_batch_size is not None:
        cfg.train.micro_batch_size = args.micro_batch_size
    if args.global_batch_size is not None:
        cfg.train.global_batch_size = args.global_batch_size
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.train_tokens_file is not None:
        cfg.train.train_tokens_file = str(Path(args.train_tokens_file).resolve())
    if args.val_tokens_file is not None:
        cfg.train.val_tokens_file = str(Path(args.val_tokens_file).resolve())
    if args.recompute_granularity is not None:
        cfg.train.recompute_granularity = (
            None
            if args.recompute_granularity.lower() in {"", "none", "null", "off", "false", "0"}
            else args.recompute_granularity
        )
    load_path = args.load or os.environ.get("MEGATRON_LOAD_CHECKPOINT") or os.environ.get("MEGATRON_RESUME_FROM")
    startup_log("init_distributed begin", all_ranks=True)
    local_rank = init_distributed(cfg)
    startup_log(
        f"init_distributed done world={torch.distributed.get_world_size()} local_rank={local_rank}",
        all_ranks=True,
    )
    torch.manual_seed(cfg.train.seed + torch.distributed.get_rank())
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    pp_size = pipeline_world_size()
    dp_size = max(1, data_parallel_world_size())
    pre_grad_accum_steps = max(
        1,
        math.ceil(cfg.train.global_batch_size / (cfg.train.micro_batch_size * dp_size)),
    )
    wandb_init_if_enabled(
        cfg,
        args.config,
        training_stack="post_distributed_pre_model",
        dp_size=dp_size,
        pp_size=pp_size,
        grad_accum_steps=pre_grad_accum_steps,
        lr_scheduler=None,
    )
    wandb_barrier_if_enabled()

    startup_log("build_model begin", all_ranks=True)
    base_model = build_model(cfg).to(device)
    sync_sequence_parallel_parameters(base_model)
    startup_log("build_model done", all_ranks=True)
    startup_log("build_training_stack begin", all_ranks=True)
    model, optimizer, training_stack = build_training_stack(base_model, cfg, local_rank, dp_size)
    baseline_profile_hooks = attach_baseline_layer_profile_hooks(model, cfg)
    lr_scheduler = build_optimizer_lr_scheduler(optimizer, cfg)
    print_dual_tp_debug(model, 0, "init", include_grads=False)
    overlap_text = (
        f" overlap_grad_reduce={getattr(model, '_dar_overlap_grad_reduce', 'na')}"
        f" overlap_param_gather={getattr(model, '_dar_overlap_param_gather', 'na')}"
    )
    startup_log(
        f"build_training_stack done stack={training_stack}{overlap_text} "
        f"baseline_layer_profile_hooks={baseline_profile_hooks}",
        all_ranks=True,
    )
    scheduler_enabled, warmup_iters, decay_iters, decay_style, warmup_init_lr = lr_scheduler_settings(cfg)
    startup_log(
        f"lr_scheduler enabled={int(scheduler_enabled)} base_lr={float(cfg.train.lr):.3e} "
        f"min_lr={float(cfg.train.min_lr):.3e} warmup_iters={warmup_iters} "
        f"warmup_init_lr={warmup_init_lr:.3e} decay_iters={decay_iters} style={decay_style} "
        f"current_lr={_optimizer_lr_text(optimizer)}",
        all_ranks=True,
    )
    loaded_step = load_checkpoint(load_path, model, optimizer, lr_scheduler) if load_path else 0
    start_step = loaded_step + 1
    if start_step > cfg.train.train_iters:
        if torch.distributed.get_rank() == 0:
            print(
                f"[checkpoint] loaded_step={loaded_step} >= train_iters={cfg.train.train_iters}; nothing to train",
                flush=True,
            )
        return

    grad_accum_steps = max(1, math.ceil(cfg.train.global_batch_size / (cfg.train.micro_batch_size * dp_size)))

    startup_log(f"dataset begin path={cfg.train.train_tokens_file}", all_ranks=True)
    dataset = PackedTokenDataset(
        cfg.train.train_tokens_file,
        cfg.model.seq_length,
        seed=cfg.train.seed,
        start_token=cfg.train.start_token,
        shuffle_buffer_size=cfg.train.shuffle_buffer_size,
    )
    startup_log(f"dataset done num_sequences={dataset.num_sequences}", all_ranks=True)
    startup_log("dataloader begin", all_ranks=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        num_workers=cfg.train.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
    )
    iterator = iter(loader)
    startup_log("dataloader done", all_ranks=True)
    val_iterator = None
    if evaluation_enabled(cfg):
        startup_log(
            f"validation dataset begin path={cfg.train.val_tokens_file} "
            f"eval_interval={cfg.train.eval_interval} eval_iters={cfg.train.eval_iters}",
            all_ranks=True,
        )
        val_dataset = PackedTokenDataset(
            cfg.train.val_tokens_file,
            cfg.model.seq_length,
            seed=cfg.train.seed + 1,
            start_token=0,
            shuffle_buffer_size=1,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.micro_batch_size,
            num_workers=cfg.train.dataloader_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_iterator = iter(val_loader)
        startup_log(f"validation dataloader done num_sequences={val_dataset.num_sequences}", all_ranks=True)

    rank = torch.distributed.get_rank()
    if rank == 0:
        tokens_per_step = cfg.train.global_batch_size * cfg.model.seq_length
        train_dtype = torch.bfloat16 if cfg.train.bf16 else torch.float16 if cfg.train.fp16 else torch.float32
        pp_bubble = estimated_pp_bubble_fraction(pp_size, grad_accum_steps)
        pp_package_text = "none"
        if pp_size > 1 and cfg.model.variant.lower() in {"attention_residual_block", "dual_attention_residual_block"}:
            recv_shape, send_shape = attnres_stage_package_shapes(cfg, cfg.train.micro_batch_size)
            pp_package_text = (
                f"recv={recv_shape}/{format_bytes(tensor_shape_nbytes(recv_shape, train_dtype))} "
                f"send={send_shape}/{format_bytes(tensor_shape_nbytes(send_shape, train_dtype))}"
            )
        print(
            f"variant={cfg.model.variant} tp={cfg.parallel.tensor_model_parallel_size} "
            f"pp={cfg.parallel.pipeline_model_parallel_size} ep={cfg.parallel.expert_model_parallel_size} dp={dp_size} "
            f"micro={cfg.train.micro_batch_size} global={cfg.train.global_batch_size} "
            f"grad_accum={grad_accum_steps} pp_bubble_est={pp_bubble * 100.0:.1f}% "
            f"tokens_per_step={tokens_per_step / 1.0e6:.2f}M seed={cfg.train.seed} "
            f"recompute={cfg.train.recompute_granularity or 'none'} "
            f"moe_dispatcher={cfg.model.moe_token_dispatcher_type} "
            f"moe_permute_fusion={getattr(cfg.model, 'moe_permute_fusion', False)} "
            f"start_token={cfg.train.start_token} shuffle_buffer={cfg.train.shuffle_buffer_size} "
            f"training_stack={training_stack}{overlap_text} pp_package={pp_package_text}"
        )
        print(
            "profiles="
            f"stage:{int(profile_enabled())} "
            f"baseline_layer:{int(baseline_layer_profile_enabled())} "
            f"dual:{int(dual_block_profile_enabled())} "
            f"attres:{int(attres_block_profile_enabled())}",
            flush=True,
        )
        tp_hidden = cfg.model.hidden_size // max(1, cfg.parallel.tensor_model_parallel_size)
        if cfg.parallel.tensor_model_parallel_size >= 4 and tp_hidden <= 512:
            print(
                "[perf_warning] TP is high for this hidden size: "
                f"hidden/TP={tp_hidden}. This can create skinny GEMMs and extra TP communication, "
                "especially with MoE all-to-all.",
                flush=True,
            )
    model.train()
    zero_training_gradients(model, optimizer, reset_grad_buffers=False)
    start = time.perf_counter()
    train_start = start
    if pp_size > 1:
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        forward_backward_func = get_forward_backward_func()
        adjust_tensor_shapes_fn = make_adjust_tensor_shapes_fn(cfg)
        for step in range(start_step, cfg.train.train_iters + 1):
            stage_profile = profile_enabled() and step % cfg.train.log_interval == 0
            stage_times: dict[str, float] = {}
            if step == 1:
                startup_log(
                    f"first_pipeline_step begin num_microbatches={grad_accum_steps}",
                    all_ranks=True,
                )
            zero_training_gradients(model, optimizer, reset_grad_buffers=(step > start_step))
            t0 = timer_now() if stage_profile else 0.0
            losses_reduced = call_with_supported_kwargs(
                forward_backward_func,
                forward_step_func=forward_step_func,
                data_iterator=[iterator],
                model=[model],
                num_microbatches=grad_accum_steps,
                seq_length=cfg.model.seq_length,
                micro_batch_size=cfg.train.micro_batch_size,
                decoder_seq_length=cfg.model.seq_length,
                forward_only=False,
                collect_non_loss_data=False,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            )
            if stage_profile:
                t1 = timer_now()
                stage_times["forward_backward"] = t1 - t0
            if step == 1:
                startup_log("first_pipeline_forward_backward done", all_ranks=True)
            t0 = timer_now() if stage_profile else 0.0
            finish_training_grad_sync(model)
            if stage_profile:
                t1 = timer_now()
                stage_times["grad_sync"] = t1 - t0
            if step == 1:
                startup_log("first_grad_sync done", all_ranks=True)
            if (not is_mcore_optimizer(optimizer)) and cfg.train.clip_grad > 0:
                if step == 1:
                    startup_log("first_clip_grad begin", all_ranks=True)
                t0 = timer_now() if stage_profile else 0.0
                clip_grad_norm_global_(model.parameters(), cfg.train.clip_grad, dp_size=dp_size)
                if stage_profile:
                    t1 = timer_now()
                    stage_times["clip_grad"] = t1 - t0
                if step == 1:
                    startup_log("first_clip_grad done", all_ranks=True)
            if step == 1:
                startup_log("first_optimizer_step begin", all_ranks=True)
            print_grad_debug(model, optimizer, step)
            print_dual_tp_debug(model, step, "post_grad_sync", include_grads=True)
            update_debug_samples = capture_update_debug(model)
            t0 = timer_now() if stage_profile else 0.0
            update_successful, optim_grad_norm, optim_num_zeros = step_training_optimizer(optimizer)
            current_lr = step_optimizer_lr_scheduler(lr_scheduler, optimizer, update_successful)
            print_dual_tp_debug(model, step, "post_optimizer", include_grads=False)
            print_update_debug(step, update_debug_samples)
            print_optimizer_debug(step, update_successful, optim_grad_norm, optim_num_zeros)
            if stage_profile:
                t1 = timer_now()
                stage_times["optimizer"] = t1 - t0
            if step == 1:
                startup_log("first_optimizer_step done", all_ranks=True)

            if step % cfg.train.log_interval == 0:
                loss = extract_scheduler_metric(losses_reduced, "loss")
                lm_loss = extract_scheduler_metric(losses_reduced, "lm_loss")
                aux_loss = extract_scheduler_metric(losses_reduced, "aux_loss")
                t0 = timer_now() if stage_profile else 0.0
                tracker_lb_loss = read_and_reset_moe_load_balancing_loss(
                    normalizer=grad_accum_steps * max(1, cfg.train.log_interval)
                )
                if stage_profile:
                    t1 = timer_now()
                    stage_times["lb_metric"] = t1 - t0
                now = time.perf_counter()
                elapsed = max(now - start, 1.0e-6)
                total_elapsed = max(now - train_start, 1.0e-6)
                tokens_per_step = cfg.train.global_batch_size * cfg.model.seq_length
                tok_s = (tokens_per_step * cfg.train.log_interval) / elapsed
                if should_log_from_rank() and loss is not None:
                    display_loss = loss
                    display_lm_loss = lm_loss if lm_loss is not None else display_loss
                    display_aux_loss = tracker_lb_loss
                    avg_step_s = total_elapsed / max(1, step)
                    eta_s = avg_step_s * max(0, cfg.train.train_iters - step)
                    aux_text = "nan" if display_aux_loss is None else f"{float(display_aux_loss):.6f}"
                    print(
                        f"{progress_bar(step, cfg.train.train_iters)} iter={step}/{cfg.train.train_iters} "
                        f"lr={current_lr} "
                        f"loss={float(display_loss):.6f} lm_loss={float(display_lm_loss):.6f} "
                        f"lb_loss={aux_text} loss_raw={float(loss):.6f} "
                        f"tok/s={tok_s:.0f} elapsed={format_duration(total_elapsed)} eta={format_duration(eta_s)}"
                    )
                    wandb_log(
                        step,
                        {
                            "train/lr": _optimizer_lr_value(optimizer),
                            "train/loss": display_loss,
                            "train/lm_loss": display_lm_loss,
                            "train/lb_loss": display_aux_loss,
                            "train/loss_raw": loss,
                            "train/aux_loss": aux_loss,
                            "train/tok_s": tok_s,
                            "train/update_successful": int(update_successful),
                            "train/grad_norm": optim_grad_norm,
                            "train/num_zeros": optim_num_zeros,
                            "train/step_seconds": elapsed / max(1, int(cfg.train.log_interval)),
                            "train/elapsed_seconds": total_elapsed,
                            "train/eta_seconds": eta_s,
                            "perf/tok_s": tok_s,
                        },
                    )
                    if stage_profile:
                        timing = " ".join(f"{name}={value:.3f}s" for name, value in stage_times.items())
                        print(f"[stage_profile] iter={step} {timing}", flush=True)
                    print_baseline_layer_profile(step)
                    print_residual_block_profiles(step)
                start = now

            if val_iterator is not None and step % cfg.train.eval_interval == 0:
                zero_training_gradients(model, optimizer, reset_grad_buffers=False)
                eval_t0 = time.perf_counter()
                val_loss, val_lb_loss = evaluate_pipeline(
                    model,
                    val_iterator,
                    cfg,
                    grad_accum_steps,
                    forward_backward_func,
                    adjust_tensor_shapes_fn,
                )
                print_validation_result(step, cfg, val_loss, val_lb_loss, time.perf_counter() - eval_t0)
                zero_training_gradients(model, optimizer, reset_grad_buffers=False)

            if cfg.train.save_interval > 0 and step % cfg.train.save_interval == 0:
                maybe_save_checkpoint(cfg.train.output_dir, model, optimizer, lr_scheduler, step)

        if val_iterator is not None and cfg.train.train_iters % cfg.train.eval_interval != 0:
            zero_training_gradients(model, optimizer, reset_grad_buffers=False)
            eval_t0 = time.perf_counter()
            val_loss, val_lb_loss = evaluate_pipeline(
                model,
                val_iterator,
                cfg,
                grad_accum_steps,
                forward_backward_func,
                adjust_tensor_shapes_fn,
            )
            print_validation_result(cfg.train.train_iters, cfg, val_loss, val_lb_loss, time.perf_counter() - eval_t0)
            zero_training_gradients(model, optimizer, reset_grad_buffers=False)

        if cfg.train.save_interval <= 0 or cfg.train.train_iters % cfg.train.save_interval != 0:
            maybe_save_checkpoint(cfg.train.output_dir, model, optimizer, lr_scheduler, cfg.train.train_iters)
        return

    set_moe_aux_loss_scale(grad_accum_steps, cfg.parallel.context_parallel_size)
    for step in range(start_step, cfg.train.train_iters + 1):
        stage_profile = profile_enabled() and step % cfg.train.log_interval == 0
        stage_times: dict[str, float] = {}
        step_loss = None
        t0 = timer_now() if stage_profile else 0.0
        for micro_idx in range(grad_accum_steps):
            sync_context = (
                model.no_sync()
                if hasattr(model, "no_sync") and micro_idx < grad_accum_steps - 1
                else nullcontext()
            )
            with sync_context:
                batch = next(iterator).to(device, non_blocking=True)
                tokens, labels = batch_to_tokens_and_labels(batch)
                attention_mask = make_left_to_right_attention_mask(tokens)
                position_ids = make_position_ids(tokens)
                with autocast_context_for_model(model):
                    out = model(tokens, labels=labels, attention_mask=attention_mask, position_ids=position_ids)
                loss = mean_loss_from_model_output(out, labels) / grad_accum_steps
                loss.backward()
                step_loss = loss.detach() if step_loss is None else step_loss + loss.detach()
        if stage_profile:
            t1 = timer_now()
            stage_times["forward_backward"] = t1 - t0
        t0 = timer_now() if stage_profile else 0.0
        finish_training_grad_sync(model)
        if stage_profile:
            t1 = timer_now()
            stage_times["grad_sync"] = t1 - t0
        if (not is_mcore_optimizer(optimizer)) and cfg.train.clip_grad > 0:
            t0 = timer_now() if stage_profile else 0.0
            clip_grad_norm_global_(model.parameters(), cfg.train.clip_grad, dp_size=dp_size)
            if stage_profile:
                t1 = timer_now()
                stage_times["clip_grad"] = t1 - t0
        print_grad_debug(model, optimizer, step)
        print_dual_tp_debug(model, step, "post_grad_sync", include_grads=True)
        update_debug_samples = capture_update_debug(model)
        t0 = timer_now() if stage_profile else 0.0
        update_successful, optim_grad_norm, optim_num_zeros = step_training_optimizer(optimizer)
        current_lr = step_optimizer_lr_scheduler(lr_scheduler, optimizer, update_successful)
        if stage_profile:
            t1 = timer_now()
            stage_times["optimizer"] = t1 - t0
        print_dual_tp_debug(model, step, "post_optimizer", include_grads=False)
        print_update_debug(step, update_debug_samples)
        print_optimizer_debug(step, update_successful, optim_grad_norm, optim_num_zeros)
        t0 = timer_now() if stage_profile else 0.0
        zero_training_gradients(model, optimizer)
        if stage_profile:
            t1 = timer_now()
            stage_times["zero_grad"] = t1 - t0

        if step % cfg.train.log_interval == 0:
            mean_loss = reduce_mean(step_loss)
            t0 = timer_now() if stage_profile else 0.0
            tracker_lb_loss = read_and_reset_moe_load_balancing_loss(
                normalizer=grad_accum_steps * max(1, cfg.train.log_interval)
            )
            if stage_profile:
                t1 = timer_now()
                stage_times["lb_metric"] = t1 - t0
            now = time.perf_counter()
            elapsed = max(now - start, 1.0e-6)
            total_elapsed = max(now - train_start, 1.0e-6)
            tokens_per_step = cfg.train.global_batch_size * cfg.model.seq_length
            tok_s = (tokens_per_step * cfg.train.log_interval) / elapsed
            if rank == 0:
                avg_step_s = total_elapsed / max(1, step)
                eta_s = avg_step_s * max(0, cfg.train.train_iters - step)
                print(
                    f"{progress_bar(step, cfg.train.train_iters)} iter={step}/{cfg.train.train_iters} "
                    f"lr={current_lr} loss={float(mean_loss):.6f} "
                    f"lb_loss={'nan' if tracker_lb_loss is None else f'{float(tracker_lb_loss):.6f}'} "
                    f"tok/s={tok_s:.0f} "
                    f"elapsed={format_duration(total_elapsed)} eta={format_duration(eta_s)}"
                )
                wandb_log(
                    step,
                    {
                        "train/lr": _optimizer_lr_value(optimizer),
                        "train/loss": mean_loss,
                        "train/lm_loss": mean_loss,
                        "train/lb_loss": tracker_lb_loss,
                        "train/tok_s": tok_s,
                        "train/update_successful": int(update_successful),
                        "train/grad_norm": optim_grad_norm,
                        "train/num_zeros": optim_num_zeros,
                        "train/step_seconds": elapsed / max(1, int(cfg.train.log_interval)),
                        "train/elapsed_seconds": total_elapsed,
                        "train/eta_seconds": eta_s,
                        "perf/tok_s": tok_s,
                    },
                )
                if stage_profile:
                    timing = " ".join(f"{name}={value:.3f}s" for name, value in stage_times.items())
                    print(f"[stage_profile] iter={step} {timing}", flush=True)
                print_baseline_layer_profile(step)
                print_residual_block_profiles(step)
            start = now

        if val_iterator is not None and step % cfg.train.eval_interval == 0:
            eval_t0 = time.perf_counter()
            val_loss, val_lb_loss = evaluate_no_pipeline(model, val_iterator, cfg, device, grad_accum_steps)
            print_validation_result(step, cfg, val_loss, val_lb_loss, time.perf_counter() - eval_t0)
            zero_training_gradients(model, optimizer, reset_grad_buffers=False)

        if cfg.train.save_interval > 0 and step % cfg.train.save_interval == 0:
            maybe_save_checkpoint(cfg.train.output_dir, model, optimizer, lr_scheduler, step)

    if val_iterator is not None and cfg.train.train_iters % cfg.train.eval_interval != 0:
        eval_t0 = time.perf_counter()
        val_loss, val_lb_loss = evaluate_no_pipeline(model, val_iterator, cfg, device, grad_accum_steps)
        print_validation_result(cfg.train.train_iters, cfg, val_loss, val_lb_loss, time.perf_counter() - eval_t0)
        zero_training_gradients(model, optimizer, reset_grad_buffers=False)

    if cfg.train.save_interval <= 0 or cfg.train.train_iters % cfg.train.save_interval != 0:
        maybe_save_checkpoint(cfg.train.output_dir, model, optimizer, lr_scheduler, cfg.train.train_iters)


if __name__ == "__main__":
    main()
