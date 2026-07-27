# Megatron 0.5B Dual-Attention Block-2 Training Design

## Goal

Add a single-node Megatron-Core training configuration for the TeleChat-style
0.5B `dual_attention_residual_block` model. The model must reproduce the
dimension flow of the existing DeepSpeed 0.5B implementation, train on the
local FineWeb-Edu 10B token files, and expose a simple eight-GPU launcher.

The name "0.5B" follows the existing DeepSpeed convention: the transformer
body is approximately 0.5B parameters. Including the tied 131,072 by 1,024
token embedding, the full model is approximately 0.63B parameters.

## Architecture

The model uses these dimensions and behaviors:

- variant: `dual_attention_residual_block`
- vocabulary size: 131,072
- sequence length: 4,096
- token embedding width: 1,024
- transformer hidden width: 1,280
- layers: 20
- attention heads: 20
- query groups / KV heads: 10
- FFN hidden width: 5,120
- dual residual streams: 2
- layers per residual block: 2, producing 10 residual blocks
- dual KV mode: cross
- dual stream mixing: `scalar_mhc_2x2`
- normalization: RMSNorm with epsilon `1e-5`
- RoPE base: 10,000
- initializer standard deviation: 0.0048
- no linear or QKV biases
- tied token embedding and output vocabulary weights

The tensor dimension flow is:

```text
token ids
  -> tied vocabulary embedding [vocab, 1024]
  -> input projection [1024, 1280]
  -> expand to two residual streams
  -> 20-layer / 10-block dual-attention transformer body
  -> residual output aggregation and stream reduction
  -> output projection [1280, 1024]
  -> RMSNorm(1024)
  -> tied vocabulary logits
```

This matches the DeepSpeed model's placement of the input and output
projections. The final normalization operates at embedding width, after the
output projection.

## Megatron Model Changes

`ModelConfig` gains an optional `embedding_size`. When omitted, it resolves to
`hidden_size`, preserving all existing configurations and checkpoints.

For `ResidualMCoreModel` when `embedding_size != hidden_size`:

1. The vocabulary embedding is created at `embedding_size`.
2. A bias-free input projection maps embeddings to `hidden_size` before stream
   expansion.
3. A bias-free output projection maps the reduced transformer output back to
   `embedding_size`.
4. The final RMSNorm and vocabulary output layer operate at `embedding_size`.
5. The vocabulary output layer shares the embedding parameter exactly as it
   does for equal-width models.

The two projections use the model initializer standard deviation. Existing
equal-width models retain identity projections and their current module shapes
and execution order.

The first version supports unequal embedding and hidden widths only with TP=1
and PP=1. This is sufficient for the requested single-node data-parallel run
and avoids claiming unsupported tensor-parallel or cross-pipeline weight tying
semantics. Configuration validation rejects an unequal-width setup outside
those constraints with an actionable error.

## Training Configuration

The new configuration uses:

- one node with eight GPUs
- TP / PP / EP / CP: 1 / 1 / 1 / 1
- data-parallel size: 8
- sequence parallelism: disabled
- precision: BF16
- selective activation recomputation
- distributed optimizer enabled
- micro batch size per GPU: 8
- global batch size: 256
- gradient accumulation: 4
- optimizer: AdamW
- peak / minimum learning rate: `3e-4` / `3e-5`
- warmup: 300 iterations
- decay: cosine over the complete training run
- Adam betas: 0.9 / 0.95
- Adam epsilon: `1e-8`
- weight decay: 0.1
- gradient clipping: 1.0
- random seed: 42
- shuffle buffer: 10,000 sequences

The YAML owns the model, optimizer, data, batch, evaluation, and checkpoint
values. The launcher exports the existing Megatron scheduler environment
variables for 300 warmup iterations, 9,537 cosine-decay iterations, and the
`3e-5` minimum learning rate; this follows the scheduler interface already used
by `train.py`.

Each optimizer step consumes `256 * 4096 = 1,048,576` tokens. Training for
10 billion requested tokens therefore uses 9,537 iterations and consumes
10,000,269,312 tokens, the smallest whole-step total at or above 10 billion.

The default data paths are:

```text
data/fineweb_edu_10BT_pretok/train.bin
data/fineweb_edu_10BT_pretok/val.bin
```

The configuration saves and evaluates every 1,000 iterations, evaluates for
20 iterations, logs every iteration, and saves a final checkpoint through the
existing training behavior.

## Launcher

Add a dedicated single-node shell launcher that follows the existing launcher
conventions and runs eight local processes with `torch.distributed.run`. It
uses the new 0.5B configuration by default and accepts the existing environment
overrides for Python, GPU count, token files, output directory, and training
iterations.

The launcher itself stays foreground-safe and composes cleanly with `nohup`.
The README documents a background invocation that writes a log and PID file,
plus commands for following the log and stopping the controller process.

## Validation

Automated tests cover:

1. loading the new configuration and checking all architecture, parallel, data,
   batch, and iteration values;
2. defaulting `embedding_size` to `hidden_size` for existing configurations;
3. rejecting unequal embedding/hidden widths when TP or PP exceeds one;
4. verifying the launcher command in dry-run mode;
5. constructing the residual model with lightweight Megatron-compatible test
   doubles and checking embedding, projection, normalization, output-layer, and
   tied-weight dimensions.

On the target GPU server, verification finishes with a one-iteration eight-GPU
smoke run using the FineWeb-Edu binaries. Success means all ranks initialize,
the printed global batch and accumulation values are correct, the loss is
finite, and the process exits normally.

## Compatibility and Scope

Existing baseline, attention-residual, dual-residual, and benchmark configs
must behave unchanged when `embedding_size` is absent. Existing user changes in
`dual_attentions_residuals/mcore_compat.py` and untracked benchmark configs are
outside this work and must not be modified.

Unequal-width TP>1, PP>1, checkpoint conversion from the DeepSpeed model, and
multi-node launch support are explicitly out of scope for this first 0.5B
configuration.
