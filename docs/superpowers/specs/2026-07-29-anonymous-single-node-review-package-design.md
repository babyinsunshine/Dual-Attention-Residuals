# Anonymous Single-Node Review Package Design

## Goal

Produce a minimal, anonymous reviewer-facing repository that reproduces the
three 7B experiments on one machine and provides a complete Hugging Face
FineWeb-Edu-100BT download and preprocessing path.

## Public Surface

The final repository exposes only these workflows:

1. Download FineWeb-Edu-100BT from Hugging Face.
2. Convert the downloaded Parquet shards into `train.bin`, `val.bin`, and
   `metadata.json`.
3. Train the Baseline, Attention Residual, or Dual Attention Residual 7B model
   through one launcher per model.
4. Run the retained correctness tests.

The README will describe those workflows directly. It will not discuss
unsupported launch topologies, prior internal workflows, authors, external
checkpoints, repositories, websites, or paper URLs.

## Repository Shape

Retain:

- `configs/7b_moe_baseline.yaml`
- `configs/7b_moe_attnres.yaml`
- `configs/7b_moe_dual.yaml`
- `scripts/train_baseline_single_node.sh`
- `scripts/train_attnres_single_node.sh`
- `scripts/train_dual_single_node.sh`
- `scripts/attnres_kernel_mode.sh`
- model, tokenizer, training, and relevant correctness-test code

Remove:

- SSH, hostfile, four-node, and generic multi-node launchers
- tests dedicated to those launchers
- the three untracked 1B benchmark configurations
- legacy planning/specification documents, including this design document in
  the final cleanup commit
- the unused overview image
- the older serial/streaming preprocessing implementation

The untracked `main.pdf` is user-owned and remains untouched. The existing
user modification in `dual_attentions_residuals/mcore_compat.py` also remains
untouched unless a repository hygiene check proves that it exposes identifying
information.

## Data Pipeline

### Download

`data/download.py` uses `huggingface_hub.snapshot_download` with these defaults:

- repository: `HuggingFaceFW/fineweb-edu`
- repository type: dataset
- included files: `sample/100BT/*.parquet`
- destination: `data/fineweb_edu_100BT`

Arguments permit overriding the destination and worker count. Hub-native
resume behavior handles interrupted downloads. The downloader does not add a
second provider or a custom cache format.

### Preprocess

The current robust parallel implementation becomes the sole public
`data/preprocess.py`. Its defaults point to the downloader output and produce
`data/fineweb_edu_100BT_pretok`. Existing resumability, atomic finalization,
input identity validation, bounded multiprocessing, and metadata behavior are
retained. The public command works without mandatory arguments; optional
arguments still allow worker-count, validation-size, file-limit, tokenizer,
and overwrite/resume control.

Data flows as follows:

`Hugging Face Parquet shards -> parallel tokenizer -> ordered shard merge -> train.bin / val.bin / metadata.json`

The merge order remains deterministic. Final files become visible only after
successful finalization; `.partial` files remain recovery state rather than
training inputs.

## Training Interface

Each of the three launchers uses `torch.distributed.run --standalone` and the
matching 7B configuration. The documented command sets only the variables a
reviewer normally needs: Python executable, GPU count, token files, and output
directory. Existing optional iteration override behavior remains available for
smoke tests.

No launcher, README section, or default configuration refers to SSH,
hostfiles, node ranks, rendezvous addresses, or cluster-specific paths.

## README and Anonymity

The README is rewritten rather than incrementally edited. It contains:

1. a short method summary;
2. tested software requirements and installation commands;
3. the two-command Hugging Face data pipeline;
4. the three training commands;
5. the test command;
6. the license statement.

It contains no Markdown links or plain-text URLs. Dataset and package names are
written as identifiers. Commands use repository-relative paths. Personal
names, usernames, local absolute paths, IP addresses, and externally hosted
artifacts are excluded.

## Error Handling

- Download errors propagate with a nonzero exit status and can be retried.
- Preprocessing rejects missing inputs, incompatible resume metadata, invalid
  worker counts, and unsafe output collisions before modifying final files.
- Training launchers retain strict shell error handling and fail immediately
  if the selected Python environment or configuration is invalid.

## Verification

Implementation is complete only after all of the following pass:

1. downloader CLI/help and mocked default-argument tests;
2. renamed parallel-preprocessor tests, including resume/finalization cases;
3. single-node launcher dry runs for all three models;
4. retained architecture, data-parallel sampling, optimizer, kernel-mode,
   residual-query, and checkpoint tests;
5. repository hygiene tests confirming:
   - exactly the three reviewer training launchers and three 7B configs remain;
   - no cluster launcher artifacts remain;
   - README has no URL or identifying path/address;
   - Hugging Face dataset, subset pattern, and output defaults are correct;
6. shell syntax checks and Python bytecode compilation;
7. a final tracked-file and dirty-worktree audit proving user-owned changes
   were not accidentally included.

No large dataset download or full training run is required for the local test
suite; network and GPU behavior are exercised through bounded smoke checks or
mocks.
