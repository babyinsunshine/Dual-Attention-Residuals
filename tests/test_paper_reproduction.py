from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import train
from dual_attentions_residuals.config import ExperimentConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_scheduler_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("MEGATRON_LR_") or name == "MEGATRON_ENABLE_LR_SCHEDULER":
            monkeypatch.delenv(name)


@pytest.mark.parametrize("variant", ["baseline", "attnres", "dual"])
def test_table11_recipe_and_schedule_are_effective(variant):
    cfg = load_config(ROOT / "configs" / f"7b_moe_{variant}.yaml")
    dp_size = 8 // cfg.parallel.tensor_model_parallel_size
    assert cfg.train.micro_batch_size == 2
    assert cfg.train.global_batch_size // (cfg.train.micro_batch_size * dp_size) == 16
    assert cfg.train.train_iters == 60000
    assert cfg.train.train_iters * cfg.train.global_batch_size * cfg.model.seq_length == 31457280000
    assert train.lr_scheduler_settings(cfg) == (True, 2000, 60000, "cosine", 0.0)


def test_scheduler_uses_config_and_preserves_environment_overrides(monkeypatch):
    cfg = ExperimentConfig()
    cfg.train.enable_lr_scheduler = True
    cfg.train.lr_warmup_iters = 100
    cfg.train.lr_decay_iters = 500
    cfg.train.lr_decay_style = "linear"
    cfg.train.lr_warmup_init_lr = 1.0e-6
    assert train.lr_scheduler_settings(cfg) == (True, 100, 500, "linear", 1.0e-6)

    monkeypatch.setenv("MEGATRON_ENABLE_LR_SCHEDULER", "0")
    monkeypatch.setenv("MEGATRON_LR_WARMUP_ITERS", "20")
    monkeypatch.setenv("MEGATRON_LR_DECAY_ITERS", "400")
    monkeypatch.setenv("MEGATRON_LR_DECAY_STYLE", "cosine")
    monkeypatch.setenv("MEGATRON_LR_WARMUP_INIT_LR", "0")
    assert train.lr_scheduler_settings(cfg) == (False, 20, 400, "cosine", 0.0)


def test_unspecified_decay_length_tracks_training_budget():
    cfg = ExperimentConfig()
    cfg.train.train_iters = 5000
    cfg.train.lr_decay_iters = None
    assert train.lr_scheduler_settings(cfg) == (True, 2000, 5000, "cosine", 0.0)


@pytest.mark.parametrize("mode, optimized", [("fused", "1"), ("reference", "0")])
def test_dual_mode_preserves_paper_semantics_and_selects_kernels(mode, optimized):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEGATRON_DUAL_")}
    env["DUAL_KERNEL_MODE"] = mode
    command = '''source "$1"
configure_dual_kernel_mode
for name in ${!MEGATRON_DUAL_@}; do
  printf '%s=%s\\n' "$name" "${!name}"
done
printf 'MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE=%s\\n' "$MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE"
'''
    result = subprocess.run(
        ["bash", "-euc", command, "bash", (ROOT / "scripts" / "dual_kernel_mode.sh").as_posix()],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE"] == optimized
    for suffix, value in {
        "KV_MODE": "cross", "AB_SOURCE": "attended", "KEEP_SOURCE": "partial",
        "CURRENT_PARTIAL_RESIDUAL": "1", "CURRENT_KEY_IMPL": "stream_norm",
        "DEFER_STREAM_MIX": "1", "EXCLUDE_RAW_HISTORY": "0", "EMBED_AS_PARTIAL": "0",
    }.items():
        assert values[f"MEGATRON_DUAL_{suffix}"] == value
    for suffix in (
        "CACHE_HISTORY_PREFIX", "CACHE_HISTORY_PREFIX_PAIR", "RECOMPUTE_HISTORY_VALUES",
        "RECOMPUTE_HISTORY_KEYS", "TRITON_HISTORY_PREFIX_PAIR", "TRITON_CURRENT_KEY_NORM",
        "FUSE_PARTIAL_KEEP_MIX", "TRITON_FUSED_RESIDUAL_READ", "TRITON_HISTORY_APPEND_PACK",
        "TE_ATTENTION_EPILOGUE",
    ):
        assert values[f"MEGATRON_DUAL_{suffix}"] == optimized


def test_dual_reference_runs_forward_and_backward_on_cpu():
    program = """
import torch
from dual_attentions_residuals.residual_blocks import DualAttentionResidualBlock
from tests.test_per_layer_residual_queries import make_config, make_layers
block = DualAttentionResidualBlock(make_config("dual_attention_residual_block"), make_layers(3), 0)
x = torch.randn(4, 2, 8, requires_grad=True)
block(x, state={}).sum().backward()
assert torch.isfinite(x.grad).all()
"""
    subprocess.run(
        ["bash", "-euc", 'source scripts/dual_kernel_mode.sh; configure_dual_kernel_mode; exec "$1" -c "$2"',
         "bash", Path(sys.executable).as_posix(), program],
        cwd=ROOT, env={**os.environ, "DUAL_KERNEL_MODE": "reference", "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )


def test_dual_mode_rejects_unknown_value():
    result = subprocess.run(
        ["bash", "-euc", 'source "$1"; configure_dual_kernel_mode', "bash",
         (ROOT / "scripts" / "dual_kernel_mode.sh").as_posix()],
        env={**os.environ, "DUAL_KERNEL_MODE": "typo"},
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "DUAL_KERNEL_MODE must be fused or reference" in result.stderr
