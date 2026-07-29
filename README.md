# Dual Attention Residuals

This repository provides matched 7B Mixture-of-Experts training configurations
for three model variants: Baseline, Attention Residual, and Dual Attention
Residuals. The variants share a common training surface so their residual
mechanisms can be evaluated consistently.

## Environment

Use Python 3.10 with CUDA-capable PyTorch. The reviewer workflow was tested
with torch 2.10.0, megatron-core 0.18.2, and transformer-engine 2.10.0.

```bash
python -m pip install torch==2.10.0 megatron-core==0.18.2 transformer-engine[pytorch]==2.10.0
python -m pip install triton numpy pyarrow pyyaml sentencepiece huggingface-hub pytest
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

## Training

Set the common environment using relative data and output paths:

```bash
export GPUS_PER_NODE=8
export TRAIN_TOKENS_FILE=data/fineweb_edu_100BT_pretok/train.bin
export VAL_TOKENS_FILE=data/fineweb_edu_100BT_pretok/val.bin
export OUTPUT_DIR=outputs/review
```

Run one retained launcher for each model variant:

```bash
bash scripts/train_baseline_single_node.sh
bash scripts/train_attnres_single_node.sh
bash scripts/train_dual_single_node.sh
```

## Tests

```bash
python -m pytest tests -q
```

## License

See the local LICENSE file.
