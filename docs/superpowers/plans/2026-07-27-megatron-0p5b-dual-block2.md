# Megatron 0.5B Dual-Attention Block-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-node eight-GPU Megatron-Core configuration and launcher for the projected TeleChat-style 0.5B `dual_attention_residual_block` model trained for at least 10 billion FineWeb-Edu tokens.

**Architecture:** Extend the residual Megatron wrapper with an optional embedding width that defaults to the transformer hidden width. Unequal-width models use a tied 1,024-wide vocabulary embedding, bias-free 1,024-to-1,280 and 1,280-to-1,024 projections around the 20-layer dual-attention body, and a final 1,024-wide RMSNorm. The first implementation intentionally supports unequal widths only for TP=1 and PP=1.

**Tech Stack:** Python 3.10+, PyTorch, Megatron-Core 0.18.x, Transformer Engine, PyYAML, `unittest`, Bash, `torch.distributed.run`.

## Global Constraints

- Preserve existing behavior when `model.embedding_size` is omitted.
- Use vocabulary size 131,072, sequence length 4,096, embedding width 1,024, hidden width 1,280, 20 layers, 20 attention heads, 10 query groups, and FFN width 5,120.
- Use `dual_attention_residual_block`, two streams, two layers per residual block, cross KV mode, and `scalar_mhc_2x2` mixing.
- Use RMSNorm epsilon `1e-5`, RoPE base 10,000, initializer standard deviation 0.0048, no linear/QKV biases, and tied embedding/output weights.
- Use one node and eight GPUs with TP/PP/EP/CP = 1/1/1/1, BF16, selective recomputation, micro batch 8, global batch 256, and gradient accumulation 4.
- Train 9,537 iterations: 10,000,269,312 tokens at 1,048,576 tokens per optimizer step.
- Use AdamW settings LR `3e-4`, minimum LR `3e-5`, betas 0.9/0.95, epsilon `1e-8`, weight decay 0.1, gradient clipping 1.0, 300 warmup iterations, and cosine decay.
- Use `data/fineweb_edu_10BT_pretok/train.bin` and `data/fineweb_edu_10BT_pretok/val.bin` by default.
- Do not modify the user's existing `dual_attentions_residuals/mcore_compat.py` change or the three untracked `configs/1b_dense_*_benchmark.yaml` files.

---

## File Map

- Modify `dual_attentions_residuals/config.py`: define and validate `embedding_size` while retaining equal-width defaults.
- Modify `dual_attentions_residuals/model.py`: create width-aware embeddings, input/output projections, final norm, output layer, and tied weights.
- Create `configs/0p5b_dense_dar_block2.yaml`: hold the complete model, parallel, optimizer, data, batch, evaluation, and checkpoint configuration.
- Create `scripts/train_0p5b_dual_single_node.sh`: launch the configuration locally and supply scheduler environment variables.
- Create `tests/test_0p5b_config.py`: test config values, defaults, and unsupported parallel combinations.
- Create `tests/test_0p5b_model_projection.py`: test projection construction, model dimensions, and exact weight tying with Megatron dependencies replaced by focused doubles.
- Create `tests/test_train_0p5b_single_node.py`: test launcher arguments and scheduler environment propagation without GPUs.
- Modify `README.md`: document foreground, smoke, background, monitoring, and resume commands.

---

### Task 1: Configuration Contract and 0.5B YAML

**Files:**
- Modify: `dual_attentions_residuals/config.py:12-45, 145-190`
- Create: `configs/0p5b_dense_dar_block2.yaml`
- Create: `tests/test_0p5b_config.py`

**Interfaces:**
- Consumes: existing `load_config(path: str | Path) -> ExperimentConfig` and `_validate_config(cfg: ExperimentConfig) -> None`.
- Produces: `ModelConfig.embedding_size: Optional[int]`, resolved to an integer by `load_config`; configuration file `configs/0p5b_dense_dar_block2.yaml`.

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_0p5b_config.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from dual_attentions_residuals.config import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs" / "0p5b_dense_dar_block2.yaml"


class Model05BConfigTests(unittest.TestCase):
    def test_complete_training_contract(self) -> None:
        cfg = load_config(CONFIG)

        self.assertEqual(cfg.model.variant, "dual_attention_residual_block")
        self.assertEqual(cfg.model.vocab_size, 131072)
        self.assertEqual(cfg.model.seq_length, 4096)
        self.assertEqual(cfg.model.embedding_size, 1024)
        self.assertEqual(cfg.model.hidden_size, 1280)
        self.assertEqual(cfg.model.ffn_hidden_size, 5120)
        self.assertEqual(cfg.model.num_layers, 20)
        self.assertEqual(cfg.model.num_attention_heads, 20)
        self.assertEqual(cfg.model.num_query_groups, 10)
        self.assertEqual(cfg.model.hyper_conn_block_layers, 2)
        self.assertEqual(cfg.model.hyper_conn_n, 2)
        self.assertEqual(cfg.model.dual_kv_mode, "cross")
        self.assertEqual(cfg.model.dual_residual_mix, "scalar_mhc_2x2")
        self.assertEqual(cfg.model.rotary_base, 10000.0)
        self.assertEqual(cfg.model.init_method_std, 0.0048)
        self.assertTrue(cfg.model.share_embeddings_and_output_weights)

        self.assertEqual(cfg.parallel.tensor_model_parallel_size, 1)
        self.assertEqual(cfg.parallel.pipeline_model_parallel_size, 1)
        self.assertEqual(cfg.parallel.expert_model_parallel_size, 1)
        self.assertEqual(cfg.parallel.context_parallel_size, 1)
        self.assertFalse(cfg.parallel.sequence_parallel)

        self.assertEqual(cfg.train.micro_batch_size, 8)
        self.assertEqual(cfg.train.global_batch_size, 256)
        self.assertEqual(cfg.train.train_iters, 9537)
        self.assertEqual(cfg.train.lr, 3.0e-4)
        self.assertEqual(cfg.train.min_lr, 3.0e-5)
        self.assertEqual(cfg.train.eval_interval, 1000)
        self.assertEqual(cfg.train.eval_iters, 20)
        self.assertEqual(cfg.train.save_interval, 1000)
        self.assertTrue(cfg.train.bf16)
        self.assertEqual(cfg.train.recompute_granularity, "selective")
        self.assertTrue(cfg.train.train_tokens_file.endswith("data/fineweb_edu_10BT_pretok/train.bin"))
        self.assertTrue(cfg.train.val_tokens_file.endswith("data/fineweb_edu_10BT_pretok/val.bin"))

    def test_missing_embedding_size_defaults_to_hidden_size(self) -> None:
        cfg = load_config(REPOSITORY / "configs" / "7b_moe_dual.yaml")
        self.assertEqual(cfg.model.embedding_size, cfg.model.hidden_size)

    def test_unequal_width_rejects_tensor_parallelism(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["parallel"]["tensor_model_parallel_size"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embedding_size.*TP=1"):
                load_config(path)

    def test_unequal_width_rejects_pipeline_parallelism(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["parallel"]["pipeline_model_parallel_size"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embedding_size.*PP=1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the intended failures**

Run:

```bash
python -m unittest tests.test_0p5b_config -v
```

Expected: failure because `configs/0p5b_dense_dar_block2.yaml` and `ModelConfig.embedding_size` do not exist.

- [ ] **Step 3: Add `embedding_size` resolution and validation**

In `ModelConfig`, add the field immediately after `hidden_size`:

```python
    embedding_size: Optional[int] = None
```

In `load_config`, immediately after constructing `ExperimentConfig`, resolve the compatibility default:

```python
    if cfg.model.embedding_size is None:
        cfg.model.embedding_size = int(cfg.model.hidden_size)
```

In `_validate_config`, after `block_variants` is defined, add:

```python
    embedding_size = int(cfg.model.embedding_size or cfg.model.hidden_size)
    if embedding_size <= 0:
        raise ValueError("model.embedding_size must be positive")
    cfg.model.embedding_size = embedding_size
    if embedding_size != cfg.model.hidden_size:
        if cfg.model.variant not in block_variants:
            raise ValueError("unequal model.embedding_size is supported only for residual block variants")
        if cfg.parallel.tensor_model_parallel_size != 1:
            raise ValueError("unequal model.embedding_size currently requires TP=1")
        if cfg.parallel.pipeline_model_parallel_size != 1:
            raise ValueError("unequal model.embedding_size currently requires PP=1")
```

- [ ] **Step 4: Add the complete training YAML**

Create `configs/0p5b_dense_dar_block2.yaml`:

```yaml
model:
  variant: dual_attention_residual_block
  seq_length: 4096
  vocab_size: 131072
  embedding_size: 1024
  hidden_size: 1280
  ffn_hidden_size: 5120
  num_layers: 20
  num_attention_heads: 20
  num_query_groups: 10
  init_method_std: 0.0048
  embedding_init_method_std: 0.0048
  hidden_dropout: 0.0
  attention_dropout: 0.0
  attention_backend: fused
  position_embedding_type: rope
  rotary_base: 10000.0
  normalization: RMSNorm
  layernorm_epsilon: 1.0e-5
  gated_linear_unit: true
  activation_func: silu
  add_bias_linear: false
  add_qkv_bias: false
  share_embeddings_and_output_weights: true
  num_moe_experts: null
  moe_router_topk: null
  hyper_conn_block_layers: 2
  hyper_conn_n: 2
  dual_residual_mix: scalar_mhc_2x2
  dual_kv_mode: cross

parallel:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  context_parallel_size: 1
  sequence_parallel: false

train:
  output_dir: outputs/0p5b_dense_dar_block2_fineweb_edu_10b
  train_tokens_file: data/fineweb_edu_10BT_pretok/train.bin
  val_tokens_file: data/fineweb_edu_10BT_pretok/val.bin
  seed: 42
  micro_batch_size: 8
  global_batch_size: 256
  train_iters: 9537
  eval_interval: 1000
  eval_iters: 20
  save_interval: 1000
  log_interval: 1
  shuffle_buffer_size: 10000
  lr: 0.0003
  min_lr: 0.00003
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  clip_grad: 1.0
  bf16: true
  fp16: false
  recompute_granularity: selective
  use_distributed_optimizer: true
  overlap_grad_reduce: true
  overlap_param_gather: true
  dataloader_workers: 2
```

- [ ] **Step 5: Run config tests**

Run:

```bash
python -m unittest tests.test_0p5b_config -v
```

Expected: all four tests pass.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add dual_attentions_residuals/config.py configs/0p5b_dense_dar_block2.yaml tests/test_0p5b_config.py
git commit -m "feat: add Megatron 0.5B dual config"
```

---

### Task 2: Width-Aware Residual Model and Tied Output

**Files:**
- Modify: `dual_attentions_residuals/model.py:245-255, 426-480, 610-685, 776-880`
- Create: `tests/test_0p5b_model_projection.py`

**Interfaces:**
- Consumes: resolved `cfg.model.embedding_size: int` from Task 1.
- Produces: `_embedding_size(cfg) -> int`, `_make_width_projection(in_features, out_features, transformer_config, init_std) -> nn.Module`, and `ResidualMCoreModel.input_projection` / `output_projection` modules.

- [ ] **Step 1: Write failing projection and construction tests**

Create `tests/test_0p5b_model_projection.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from dual_attentions_residuals.config import load_config
from dual_attentions_residuals import model as model_module


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs" / "0p5b_dense_dar_block2.yaml"


class FakeAtomicLayer(nn.Module):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()


class FakeBlock(nn.Module):
    def __init__(self, _cfg, _layers, **_kwargs) -> None:
        super().__init__()


class FakeAggregation(nn.Module):
    def __init__(self, _cfg) -> None:
        super().__init__()


class ModelProjectionTests(unittest.TestCase):
    def test_width_projection_shapes_and_identity(self) -> None:
        transformer_config = SimpleNamespace(params_dtype=torch.float32)
        identity = model_module._make_width_projection(1280, 1280, transformer_config, 0.0048)
        projected = model_module._make_width_projection(1024, 1280, transformer_config, 0.0048)

        self.assertIsInstance(identity, nn.Identity)
        self.assertIsInstance(projected, nn.Linear)
        self.assertEqual(tuple(projected.weight.shape), (1280, 1024))
        self.assertIsNone(projected.bias)

    def test_residual_model_uses_projected_tied_embedding_width(self) -> None:
        cfg = load_config(CONFIG)
        transformer_config = SimpleNamespace(
            params_dtype=torch.float32,
            hidden_size=1280,
            layernorm_epsilon=1.0e-5,
        )
        embedding = nn.Embedding(cfg.model.vocab_size, cfg.model.embedding_size)
        output = nn.Linear(cfg.model.embedding_size, cfg.model.vocab_size, bias=False)

        with (
            patch.object(model_module, "require_megatron_core"),
            patch.object(model_module, "_make_transformer_config", return_value=transformer_config),
            patch.object(model_module, "_make_layer_spec", return_value=object()),
            patch.object(model_module, "_make_vocab_embedding", return_value=embedding),
            patch.object(model_module, "_make_output_layer", return_value=output),
            patch.object(
                model_module,
                "_make_final_layernorm",
                side_effect=lambda _config, hidden_size=None: nn.LayerNorm(hidden_size),
            ),
            patch.object(model_module, "_make_rotary_embedding", return_value=None),
            patch.object(model_module, "MCoreAtomicLayer", FakeAtomicLayer),
            patch.object(model_module, "DualAttentionResidualBlock", FakeBlock),
            patch.object(model_module, "DualAttentionResidualOutputAggregation", FakeAggregation),
        ):
            model = model_module.ResidualMCoreModel(cfg, pre_process=True, post_process=True)

        self.assertEqual(tuple(model.embedding.weight.shape), (131072, 1024))
        self.assertEqual(tuple(model.input_projection.weight.shape), (1280, 1024))
        self.assertEqual(tuple(model.output_projection.weight.shape), (1024, 1280))
        self.assertEqual(tuple(model.final_norm.normalized_shape), (1024,))
        self.assertEqual(tuple(model.output_layer.weight.shape), (131072, 1024))
        self.assertIs(model.output_layer.weight, model.embedding.weight)
        self.assertEqual(len(model.blocks), 10)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the model test and confirm it fails**

Run:

```bash
python -m unittest tests.test_0p5b_model_projection -v
```

Expected: failure because `_make_width_projection`, `input_projection`, and `output_projection` do not exist.

- [ ] **Step 3: Add width helpers and make final norm width-aware**

Add near `_make_final_layernorm` in `dual_attentions_residuals/model.py`:

```python
def _embedding_size(cfg: ExperimentConfig) -> int:
    return int(cfg.model.embedding_size or cfg.model.hidden_size)


def _make_width_projection(
    in_features: int,
    out_features: int,
    transformer_config,
    init_std: float,
) -> nn.Module:
    if in_features == out_features:
        return nn.Identity()
    projection = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(projection.weight, mean=0.0, std=float(init_std))
    return projection.to(dtype=transformer_config.params_dtype)
```

Change `_make_final_layernorm` to:

```python
def _make_final_layernorm(transformer_config, hidden_size: Optional[int] = None) -> nn.Module:
    from megatron.core.transformer.transformer_block import LayerNormImpl

    return call_with_supported_kwargs(
        LayerNormImpl,
        config=transformer_config,
        hidden_size=hidden_size or transformer_config.hidden_size,
        eps=transformer_config.layernorm_epsilon,
    )
```

- [ ] **Step 4: Construct projections and embedding-width output modules**

In `ResidualMCoreModel.__init__`, after assigning `self.position_embedding_type`, add:

```python
        self.embedding_size = _embedding_size(cfg)
```

Replace output module construction with:

```python
        self.embedding = _make_vocab_embedding(cfg, self.transformer_config) if self.is_first_stage else None
        self.input_projection = (
            _make_width_projection(
                self.embedding_size,
                cfg.model.hidden_size,
                self.transformer_config,
                cfg.model.init_method_std,
            )
            if self.is_first_stage
            else nn.Identity()
        )
        self.output_projection = (
            _make_width_projection(
                cfg.model.hidden_size,
                self.embedding_size,
                self.transformer_config,
                cfg.model.init_method_std,
            )
            if self.is_last_stage
            else nn.Identity()
        )
        self.output_layer = _make_output_layer(cfg, self.transformer_config) if self.is_last_stage else None
        self.final_norm = (
            _make_final_layernorm(self.transformer_config, self.embedding_size)
            if self.is_last_stage
            else None
        )
```

In `_make_vocab_embedding`, use `VocabParallelEmbedding` directly for unequal widths before the current `LanguageModelEmbedding` branch:

```python
    embedding_size = _embedding_size(cfg)
    if embedding_size != cfg.model.hidden_size:
        from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

        emb = call_with_supported_kwargs(
            VocabParallelEmbedding,
            num_embeddings=cfg.model.vocab_size,
            embedding_dim=embedding_size,
            config=transformer_config,
            init_method=lambda x: torch.nn.init.normal_(
                x,
                mean=0.0,
                std=float(cfg.model.embedding_init_method_std or cfg.model.init_method_std),
            ),
        )
        setattr(emb, "_dar_raw_vocab_embedding", True)
        return emb
```

In the existing `VocabParallelEmbedding` fallback, replace
`embedding_dim=cfg.model.hidden_size` with:

```python
            embedding_dim=_embedding_size(cfg),
```

Replace `_make_output_layer` with:

```python
def _make_output_layer(cfg: ExperimentConfig, transformer_config) -> nn.Module:
    embedding_size = _embedding_size(cfg)
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        return call_with_supported_kwargs(
            ColumnParallelLinear,
            input_size=embedding_size,
            output_size=cfg.model.vocab_size,
            config=transformer_config,
            init_method=lambda x: torch.nn.init.normal_(
                x,
                mean=0.0,
                std=cfg.model.init_method_std,
            ),
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
    except Exception:
        head = nn.Linear(embedding_size, cfg.model.vocab_size, bias=False)
        nn.init.normal_(head.weight, mean=0.0, std=cfg.model.init_method_std)
        return head
```

- [ ] **Step 5: Insert projections in the exact DeepSpeed execution order**

In `ResidualMCoreModel.forward`, immediately after `_call_embedding`, add:

```python
            hidden_states = self.input_projection(hidden_states)
```

After `_reduce_streams`, replace the current dtype/norm sequence with:

```python
        hidden_states = hidden_states.to(dtype=self.transformer_config.params_dtype)
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.final_norm(hidden_states).contiguous()
```

Keep stream aggregation and stream reduction before `output_projection`, and
keep `_call_output_layer` after the final norm.

- [ ] **Step 6: Run focused model and configuration tests**

Run:

```bash
python -m unittest tests.test_0p5b_model_projection tests.test_0p5b_config -v
```

Expected: all tests pass, including exact identity tying between the embedding and output layer weights.

- [ ] **Step 7: Commit width-aware model support**

```bash
git add dual_attentions_residuals/model.py tests/test_0p5b_model_projection.py
git commit -m "feat: support projected residual embeddings"
```

---

### Task 3: Single-Node Launcher and Scheduler Propagation

**Files:**
- Create: `scripts/train_0p5b_dual_single_node.sh`
- Create: `tests/test_train_0p5b_single_node.py`

**Interfaces:**
- Consumes: `configs/0p5b_dense_dar_block2.yaml` from Task 1 and `train.py` scheduler variables.
- Produces: a foreground-safe launcher with `PYTHON_BIN`, `GPUS_PER_NODE`, `TRAIN_TOKENS_FILE`, `VAL_TOKENS_FILE`, `OUTPUT_DIR`, `TRAIN_ITERS`, `MICRO_BATCH_SIZE`, and `GLOBAL_BATCH_SIZE` overrides.

- [ ] **Step 1: Write the failing launcher test**

Create `tests/test_train_0p5b_single_node.py`:

```python
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "train_0p5b_dual_single_node.sh"


class Model05BLauncherTests(unittest.TestCase):
    def test_launcher_propagates_command_and_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            fake_python = root / "python"
            fake_python.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURE_FILE"]).write_text(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}))
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update({"PYTHON_BIN": str(fake_python), "CAPTURE_FILE": str(capture)})

            completed = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=REPOSITORY,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            argv = payload["argv"]
            child_env = payload["env"]
            self.assertEqual(argv[:4], ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"])
            self.assertIn("configs/0p5b_dense_dar_block2.yaml", argv)
            self.assertEqual(child_env["MEGATRON_ENABLE_LR_SCHEDULER"], "1")
            self.assertEqual(child_env["MEGATRON_LR_WARMUP_ITERS"], "300")
            self.assertEqual(child_env["MEGATRON_LR_DECAY_ITERS"], "9537")
            self.assertEqual(child_env["MEGATRON_LR_DECAY_STYLE"], "cosine")
            self.assertEqual(child_env["MEGATRON_LR_WARMUP_INIT_LR"], "0.0")

    def test_launcher_accepts_smoke_overrides(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPOSITORY,
            env={**os.environ, "DRY_RUN": "1", "PYTHON_BIN": "/opt/python", "GPUS_PER_NODE": "2", "TRAIN_ITERS": "1", "MICRO_BATCH_SIZE": "2", "GLOBAL_BATCH_SIZE": "4"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("/opt/python", completed.stdout)
        self.assertIn("--nproc-per-node=2", completed.stdout)
        self.assertIn("--train-iters 1", completed.stdout)
        self.assertIn("--micro-batch-size 2", completed.stdout)
        self.assertIn("--global-batch-size 4", completed.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the launcher test and confirm it fails**

Run:

```bash
python -m unittest tests.test_train_0p5b_single_node -v
```

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Add the launcher**

Create `scripts/train_0p5b_dual_single_node.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_ENABLE_LR_SCHEDULER="${MEGATRON_ENABLE_LR_SCHEDULER:-1}"
export MEGATRON_LR_WARMUP_ITERS="${MEGATRON_LR_WARMUP_ITERS:-300}"
export MEGATRON_LR_DECAY_ITERS="${MEGATRON_LR_DECAY_ITERS:-9537}"
export MEGATRON_LR_DECAY_STYLE="${MEGATRON_LR_DECAY_STYLE:-cosine}"
export MEGATRON_LR_WARMUP_INIT_LR="${MEGATRON_LR_WARMUP_INIT_LR:-0.0}"

args=(--config configs/0p5b_dense_dar_block2.yaml)
[[ -n ${TRAIN_TOKENS_FILE:-} ]] && args+=(--train-tokens-file "$TRAIN_TOKENS_FILE")
[[ -n ${VAL_TOKENS_FILE:-} ]] && args+=(--val-tokens-file "$VAL_TOKENS_FILE")
[[ -n ${OUTPUT_DIR:-} ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -n ${TRAIN_ITERS:-} ]] && args+=(--train-iters "$TRAIN_ITERS")
[[ -n ${MICRO_BATCH_SIZE:-} ]] && args+=(--micro-batch-size "$MICRO_BATCH_SIZE")
[[ -n ${GLOBAL_BATCH_SIZE:-} ]] && args+=(--global-batch-size "$GLOBAL_BATCH_SIZE")

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc-per-node="${GPUS_PER_NODE:-8}"
  train.py
  "${args[@]}"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
```

Make it executable:

```bash
chmod +x scripts/train_0p5b_dual_single_node.sh
```

- [ ] **Step 4: Run launcher tests and shell validation**

Run:

```bash
bash -n scripts/train_0p5b_dual_single_node.sh
python -m unittest tests.test_train_0p5b_single_node -v
DRY_RUN=1 PYTHON_BIN=.venv-train/bin/python bash scripts/train_0p5b_dual_single_node.sh
```

Expected: syntax check and both tests pass; dry run prints eight processes and the new config.

- [ ] **Step 5: Commit the launcher**

```bash
git add scripts/train_0p5b_dual_single_node.sh tests/test_train_0p5b_single_node.py
git commit -m "feat: launch 0.5B dual training"
```

---

### Task 4: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md:125-180`

**Interfaces:**
- Consumes: the config and launcher from Tasks 1 and 3.
- Produces: copyable smoke, foreground, background, monitoring, and resume commands.

- [ ] **Step 1: Add the documented commands**

Add a `0.5B dual-attention block-2` subsection under `### Train` in `README.md`:

````markdown
#### 0.5B dual-attention block-2

The projected 0.5B configuration matches the DeepSpeed architecture with a
1,024-wide tied embedding, a 1,280-wide transformer body, 20 layers, and two
layers per dual-attention residual block. On one eight-GPU node, micro batch 8
and global batch 256 produce four gradient-accumulation steps. The default
9,537 optimizer iterations consume 10,000,269,312 tokens.

Run a one-iteration environment and model smoke test:

```bash
PYTHON_BIN="$PWD/.venv-train/bin/python" \
TRAIN_ITERS=1 \
OUTPUT_DIR=outputs/0p5b_dense_dar_block2_smoke \
bash scripts/train_0p5b_dual_single_node.sh
```

Run the full 9,537-iteration job in the foreground:

```bash
PYTHON_BIN="$PWD/.venv-train/bin/python" \
bash scripts/train_0p5b_dual_single_node.sh
```

Start the full run in the background:

```bash
mkdir -p logs
nohup env \
  PYTHON_BIN="$PWD/.venv-train/bin/python" \
  OUTPUT_DIR=outputs/0p5b_dense_dar_block2_fineweb_edu_10b \
  bash scripts/train_0p5b_dual_single_node.sh \
  > logs/0p5b_dense_dar_block2.log 2>&1 < /dev/null &
echo $! > logs/0p5b_dense_dar_block2.pid
```

Monitor it:

```bash
tail -f logs/0p5b_dense_dar_block2.log
cat logs/0p5b_dense_dar_block2.pid
```

Resume from the latest checkpoint recorded in the output directory:

```bash
MEGATRON_RESUME_FROM=outputs/0p5b_dense_dar_block2_fineweb_edu_10b \
PYTHON_BIN="$PWD/.venv-train/bin/python" \
bash scripts/train_0p5b_dual_single_node.sh
```
````

- [ ] **Step 2: Run the complete CPU-side regression suite**

Run:

```bash
python -m unittest \
  tests.test_0p5b_config \
  tests.test_0p5b_model_projection \
  tests.test_train_0p5b_single_node \
  tests.test_launch_baseline_ssh \
  -v
python -m unittest tests.test_preprocess_parallel -v
bash -n scripts/train_0p5b_dual_single_node.sh
git diff --check
```

Expected: all tests pass, Bash syntax passes, and `git diff --check` emits no output.

- [ ] **Step 3: Confirm config math without initializing GPUs**

Run:

```bash
python - <<'PY'
from dual_attentions_residuals.config import load_config

cfg = load_config("configs/0p5b_dense_dar_block2.yaml")
tokens_per_step = cfg.train.global_batch_size * cfg.model.seq_length
total_tokens = tokens_per_step * cfg.train.train_iters
grad_accum = cfg.train.global_batch_size // (cfg.train.micro_batch_size * 8)
print("tokens_per_step:", tokens_per_step)
print("total_tokens:", total_tokens)
print("gradient_accumulation:", grad_accum)
assert tokens_per_step == 1_048_576
assert total_tokens == 10_000_269_312
assert grad_accum == 4
PY
```

Expected: all assertions pass.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain 0.5B dual training"
```

- [ ] **Step 5: Run the target-server eight-GPU smoke test**

On the training server:

```bash
cd /gemini/space/internship/yuxingda/dual_attentions_residuals_megatron
source .venv-train/bin/activate

set -o pipefail
PYTHON_BIN="$PWD/.venv-train/bin/python" \
TRAIN_ITERS=1 \
OUTPUT_DIR=outputs/0p5b_dense_dar_block2_smoke \
bash scripts/train_0p5b_dual_single_node.sh \
  2>&1 | tee 0p5b_dense_dar_block2_smoke.log
```

Expected rank-zero output includes:

```text
variant=dual_attention_residual_block tp=1 pp=1 ep=1 dp=8 micro=8 global=256 grad_accum=4
lr_scheduler enabled=1 base_lr=3.000e-04 min_lr=3.000e-05 warmup_iters=300
```

All eight ranks must initialize, iteration 1 must report a finite loss, and the launcher must exit with status zero.

---

## Final Review Checklist

- [ ] Existing configs resolve `embedding_size == hidden_size` and retain identity projections.
- [ ] The 0.5B config builds exactly 10 two-layer residual blocks.
- [ ] Embedding/output weights are the same `Parameter`, not copied tensors.
- [ ] Projection, final norm, and vocabulary layer dimensions match the DeepSpeed architecture.
- [ ] Scheduler reports 300 warmup and 9,537 cosine-decay iterations.
- [ ] Eight-GPU batch math reports micro 8, global 256, and accumulation 4.
- [ ] Total scheduled tokens equal 10,000,269,312.
- [ ] Existing user-modified and untracked files remain untouched.
