# Dual Attention Residuals

This repository provides matched 7B Mixture-of-Experts training configurations
for three model variants: Baseline, Attention Residual, and Dual Attention
Residuals. The variants share a common training surface so their residual
mechanisms can be evaluated consistently.

## Environment

Use Linux and Python 3.10 with CUDA-capable PyTorch. The reviewer workflow was tested
with torch 2.10.0, megatron-core 0.16.1, and transformer-engine 2.10.0.

```bash
python -m pip install torch==2.10.0 megatron-core==0.16.1 transformer-engine[pytorch]==2.10.0
python -m pip install triton numpy pyarrow pyyaml sentencepiece huggingface-hub pytest tomli
```

## Data

Download the FineWeb-Edu 100BT Parquet shards, then preprocess them:

```bash
python data/download.py
python data/preprocess.py
```

The downloader uses the default Hugging Face identifier
`HuggingFaceFW/fineweb-edu`, selects the `sample/100BT/*.parquet` pattern, and
writes raw files to `data/fineweb_edu_100BT`. Use `--num-workers` to adjust
download concurrency.

The preprocessor reads the default raw directory and writes `train.bin`,
`val.bin`, and `metadata.json` to `data/fineweb_edu_100BT_pretok`. Use
`--num-workers` to set preprocessing workers and `--resume` to continue a
previous run. It appends an EOS token by default so adjacent documents remain
separated; use `--no-add-eos` only for controlled experiments. The EOS setting
is part of the resume identity, so a run cannot resume with a different setting.
For a bounded smoke run, limit the input with `--max-files`:

```bash
python data/preprocess.py --max-files 2 --num-workers 2
```

## Paper settings (Tables 5 and 11)

The three YAML files in `configs/` use the sparse-MoE recipe in Table 11:

| Setting | Value |
| --- | --- |
| Hardware | One node, eight H800 GPUs |
| Layers / hidden size / block size | 24 / 2048 / 3 |
| TP / PP / EP / CP | 2 / 1 / 4 / 1 |
| Sequence length | 4096 |
| Micro / global batch size | 2 / 128 |
| Gradient accumulation | 16 steps (data-parallel size 4) |
| Training iterations / tokens | 60,000 / 31,457,280,000 |
| Peak / minimum learning rate | 3e-4 / 3e-5 |
| Schedule | 2,000-step warmup from zero; cosine decay to step 60,000 |
| AdamW betas / epsilon / weight decay | (0.9, 0.95) / 1e-8 / 0.1 |
| Gradient clipping / seed | 1.0 / 42 |
| Precision / activation recomputation | BF16 / selective |
| Distributed optimizer / sequence parallelism | Enabled / enabled |

The previous 100,000-iteration, micro-batch-4 defaults were reference settings;
the defaults now match the paper. The learning-rate scheduler is configured in
YAML and enabled by default, including when invoking `train.py` directly.

## Training commands

Set the common environment using relative data and output paths:

```bash
export GPUS_PER_NODE=8
export TRAIN_TOKENS_FILE=data/fineweb_edu_100BT_pretok/train.bin
export VAL_TOKENS_FILE=data/fineweb_edu_100BT_pretok/val.bin
export MEGATRON_ENABLE_LR_SCHEDULER=1
export MEGATRON_LR_WARMUP_ITERS=2000
export MEGATRON_LR_DECAY_ITERS=60000
export MEGATRON_LR_DECAY_STYLE=cosine
export MEGATRON_LR_WARMUP_INIT_LR=0
```

Run one retained launcher for each model variant:

```bash
TRAIN_ITERS=60000 OUTPUT_DIR=outputs/paper_baseline bash scripts/train_baseline_single_node.sh
TRAIN_ITERS=60000 OUTPUT_DIR=outputs/paper_attnres ATTRES_KERNEL_MODE=fused bash scripts/train_attnres_single_node.sh
TRAIN_ITERS=60000 OUTPUT_DIR=outputs/paper_dual DUAL_KERNEL_MODE=fused bash scripts/train_dual_single_node.sh
```

`TRAIN_TOKENS_FILE`, `VAL_TOKENS_FILE`, `OUTPUT_DIR`, and `TRAIN_ITERS` override
the corresponding YAML fields through the launchers. `MEGATRON_ENABLE_LR_SCHEDULER`
and `MEGATRON_LR_*` variables override the YAML schedule; the commands above set
them explicitly. Each run prints its effective schedule at startup. Add
`DRY_RUN=1` before a launcher to print its command without starting training.

To resume training, set `MEGATRON_LOAD_CHECKPOINT` to the existing output
directory and run the same launcher. New checkpoints record the next training
sample position. For older checkpoints, use the original batch and worker
settings so the position can be reconstructed from the saved step.

Both residual launchers default to fused execution. The complete kernel
switches are in `scripts/attnres_kernel_mode.sh` and
`scripts/dual_kernel_mode.sh`. Set the corresponding mode to `reference` for
the unfused path. Both DAR modes retain cross-stream keys, self-stream values,
the initial embedding history entry, and transport of the current partial
state. Fused execution requires CUDA and Triton.

Table 5 uses the same training recipe with fused residual-history operators.
Its throughput is the total across all eight GPUs, excluding the first 300
steps; peak memory is reported per GPU. Each optimizer step processes 524,288
tokens. The table reports measured performance on H800 hardware, which must be
measured again on the target machine; changing to reference kernels changes
the efficiency comparison.

## Residual parameter accounting

Queries are independent for every attention and FFN branch. DAR's gate
parameters are also independent per branch; the block size controls history
storage and does not tie these parameters. Both methods include a final
output query. For 24 layers and hidden size 2048:

| Residual mechanism | Added parameters over the common backbone |
| --- | ---: |
| AttnRes | (2 x 24 + 1) x 2048 = 100,352 |
| DAR | 2 x 24 x (12 x 2048 + 8) + 2 x 2048 = 1,184,128 |
| DAR minus AttnRes | 1,083,776 (approximately 1.084M) |

These counts exclude the common Transformer and tied vocabulary parameters.
TP/EP replicas must not be counted as additional logical model parameters.

## Tests

```bash
python -m pytest tests -q
```

## License

See the local LICENSE file.
