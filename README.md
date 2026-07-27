# Dual Attention Residuals

<p align="center">
  <img src="assets/dar_overview.png" alt="Dual Attention Residual architecture" width="100%">
</p>

<p align="center">
  <strong>A compact Megatron-Core implementation of dual attention residuals for large-scale MoE training.</strong>
</p>

Dual Attention Residuals (DAR) maintain a dual residual representation and
learn to read from its shared depth history before each Transformer block. This
repository provides matched Baseline, Attention Residual, and DAR training
paths so that only the residual mechanism changes across comparisons.

## Highlights

- Matched 7B Mixture-of-Experts configurations for all three residual variants.
- Optimized Triton operators with reference PyTorch paths for verification.
- Ready-to-use single-node and multi-node launchers built on Megatron-Core.

## Checkpoint

DAR weights are available on Hugging Face Hub as
[`moddells/dual-attentions-residuals-60k`](https://huggingface.co/moddells/dual-attentions-residuals-60k).

## Quick start

### Install

Use a generic Linux NVIDIA server and Python **3.10 exactly**: this project
declares `>=3.10,<3.11`. The primary workflow uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The repository pins `megatron-core==0.16.1`; do not install a latest 0.18.x
release over it. In particular, Megatron-Core 0.18.2 was observed to fail at
import time with an `nvidia-resiliency-ext` mismatch.

The CUDA version reported by `nvidia-smi` is the NVIDIA driver's capability,
not necessarily the CUDA runtime bundled with the installed PyTorch wheel.
For example, a driver reporting CUDA 13.2 can run a compatible wheel with a
different runtime version; the driver still must be new enough for that wheel.
After installation, verify the training stack with:

```bash
python -c "import torch, megatron.core, transformer_engine.pytorch, triton; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

For data conversion only, avoid installing Transformer Engine or Megatron-Core:

```bash
uv pip install numpy==1.26.4 pyarrow==21.0.0 sentencepiece==0.2.1
```

If Transformer Engine has no usable prebuilt wheel for the server, use the
[NVIDIA Transformer Engine source repository](https://github.com/NVIDIA/TransformerEngine)
only after PyTorch and its source-build prerequisites (a compiler, CUDA toolkit
with `nvcc`, and CUDA headers) are available:

```bash
NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation \
  'git+https://github.com/NVIDIA/TransformerEngine.git@v2.10'
```

This is a source-build fallback; `v2.10` corresponds to this repository's
pinned `transformer-engine[pytorch]==2.10.0`. A CUDA toolkit and `nvcc` are not
needed when using a compatible prebuilt PyTorch/Transformer Engine wheel.

### Prepare data

The loader consumes flat `int32` token files. The included preprocessing
command streams a public corpus and creates training and validation binaries:

```bash
python data/preprocess.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --split train \
  --tokenizer tokenizer/tokenizer.model \
  --output-dir data/fineweb_edu_pretok \
  --validation-tokens 10000000 \
  --add-eos
```

### Local FineWeb-Edu 100BT Parquet preprocessing

For a two-file smoke run, use `--max-files 2` and a small validation split:

```bash
python data/preprocess_parallel.py \
  --input-glob '/path/to/fineweb-100B/sample/100BT/*.parquet' \
  --tokenizer tokenizer/tokenizer.model \
  --output-dir data/processed_smoke \
  --num-workers 2 \
  --max-files 2 \
  --validation-tokens 100000 \
  --add-eos
```

For the full local dataset, this uses 64 workers on a 96-core server; adjust
`--num-workers` for the machine. The default is `min(64, cpu_count)`:

```bash
python data/preprocess_parallel.py \
  --input-glob '/path/to/fineweb-100B/sample/100BT/*.parquet' \
  --tokenizer tokenizer/tokenizer.model \
  --output-dir data/processed \
  --num-workers 64 \
  --validation-tokens 10000000 \
  --add-eos \
  --resume
```

Expect about 400 GB of output; reserve at least 450 GiB of free space. The
converter emits a heartbeat every 30 seconds. Re-running with identical
arguments and `--resume` deterministically continues the same job. `--resume`
may be supplied from the first invocation: it reuses completed per-file shards
before a merge journal exists, then resumes journaled merging afterward. Source
Parquet files are read-only throughout. A successful output directory contains
`train.bin`, `val.bin`, and completion marker `metadata.json`; the validation
file has exactly 10,000,000 tokens in the full command. Safe final-output
publication requires Linux support for `renameat2(RENAME_NOREPLACE)`.

### Train

```bash
bash scripts/train_baseline_single_node.sh
bash scripts/train_attnres_single_node.sh
bash scripts/train_dual_single_node.sh
```

For multi-node runs, use the corresponding `*_multi_node.sh` launcher. Dataset,
output, and schedule settings can be overridden without editing the scripts:

```bash
TRAIN_TOKENS_FILE=data/fineweb_edu_pretok/train.bin \
VAL_TOKENS_FILE=data/fineweb_edu_pretok/val.bin \
OUTPUT_DIR=outputs/dual_smoke \
TRAIN_ITERS=1000 \
bash scripts/train_dual_single_node.sh
```

For the default four-node baseline run, the controller node can launch all four
SSH sessions with one script. Its defaults are the four documented cluster IPs,
8 GPUs per node, 60,000 iterations, micro/global batch sizes 4/128, 2,000 warmup
iterations, cosine decay, and validation/checkpoint intervals of 10,000:

```bash
DRY_RUN=1 bash scripts/launch_baseline_ssh.sh

nohup bash scripts/launch_baseline_ssh.sh \
  > baseline_4node_launcher.log 2>&1 < /dev/null &
echo $! > baseline_4node_launcher.pid
```

The controller process must remain alive because it owns the SSH sessions.
Monitor the controller and rank-zero logs with:

```bash
tail -f baseline_4node_launcher.log
tail -f logs/baseline_4node_60k/baseline_node0.log
```

NCCL network selection is recorded during initialization. `NET/IB` indicates
the RDMA path; `NET/Socket` indicates TCP fallback:

```bash
grep -Eh 'NET/IB|NET/Socket|Using network|mlx5' \
  logs/baseline_4node_60k/baseline_node*.log | head -100
```

All launcher settings can be overridden through environment variables. For a
short four-node smoke test:

```bash
TRAIN_ITERS=10 \
WARMUP_ITERS=0 \
OUTPUT_DIR=outputs/baseline_4node_smoke \
LOG_DIR=logs/baseline_4node_smoke \
bash scripts/launch_baseline_ssh.sh
```

Set `HOSTS` to a whitespace-separated, node-rank-ordered list to use another
cluster. `MASTER_ADDR` defaults to the first host.

### Three independent 7B four-node runs

The Baseline, Attention Residual, and DAR jobs use separate hostfiles at the
repository root: `hostfile_baseline`, `hostfile_attnres`, and `hostfile_dar`.
Each file must contain exactly four unique hosts in node-rank order, one per
line. Blank lines and comments are allowed; the first host is rank zero and the
rendezvous master. For example:

```text
10.0.0.1
10.0.0.2
10.0.0.3
10.0.0.4
```

Before launching, finalize both token files at
`data/fineweb_100BT_pretok/train.bin` and
`data/fineweb_100BT_pretok/val.bin`. Partial files are rejected, as is a
non-empty output directory. Start each supervised controller with:

```bash
bash scripts/train_7b_baseline_4node.sh start
bash scripts/train_7b_attnres_4node.sh start
bash scripts/train_7b_dar_4node.sh start
```

Each model has independent output, controller, PID, and per-rank log paths.
Use the same entry script for lifecycle operations:

```bash
bash scripts/train_7b_dar_4node.sh status
bash scripts/train_7b_dar_4node.sh logs
bash scripts/train_7b_dar_4node.sh stop
```

At global batch size 128 and sequence length 4,096, each iteration processes
`128 * 4,096 = 524,288` token positions. The 40,000-iteration schedule
therefore processes `524,288 * 40,000 = 20,971,520,000` token positions
(20.97B). Before any production start, run a one-iteration smoke test on each
four-node group with a separate empty `OUTPUT_DIR`; do not reuse a production
output directory for the smoke test.

## Default configuration

| Setting | Value |
|---|---:|
| GPUs per node | 8 |
| Tensor / expert parallelism | 2 / 4 |
| Micro / global batch size | 4 / 128 |
| Sequence length | 4096 |
| Experts / active experts | 64 / 8 |
| Precision | BF16 |
| Sequence parallelism | Enabled |
| Activation recomputation | Selective |
| Residual operators | Optimized Triton |

## License

Released under the [MIT License](LICENSE).
