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

If Transformer Engine has no usable prebuilt wheel for the server, follow its
published source-build prerequisites instead of mixing incompatible wheels. A
CUDA toolkit and `nvcc` are needed only for that source-build fallback, not for
using a compatible prebuilt PyTorch/Transformer Engine wheel.

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
arguments and `--resume` deterministically continues the same job. Source
Parquet files are read-only throughout. A successful output directory contains
`train.bin`, `val.bin`, and completion marker `metadata.json`; the validation
file has exactly 10,000,000 tokens in the full command. Safe final publication
requires Linux support for `renameat2(RENAME_NOREPLACE)`.

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
