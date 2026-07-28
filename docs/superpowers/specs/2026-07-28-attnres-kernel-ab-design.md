# AttnRes Kernel A/B Design

## Goal

Determine whether the AttnRes training-loss regression is caused by its
Triton/fused implementation. Run two otherwise identical 10-step H100 training
jobs and compare their per-step results.

## Approaches considered

1. **Explicit launcher mode (selected).** Add
   `ATTRES_KERNEL_MODE=fused|reference`. The launcher propagates the selected
   mode to every rank; `reference` disables every AttnRes Triton, fused epilogue,
   cached-prefix, and recomputation optimization. This is auditable and prevents
   ranks from silently using different modes.
2. **Manual environment overrides.** Export all existing low-level flags before
   launching. This requires a long, error-prone command and the current SSH
   launcher does not propagate arbitrary variables.
3. **Shadow both implementations in one graph.** Compare fused and reference
   operations during one training run. This changes memory use and execution
   order, so it is less representative and unsuitable for the first diagnosis.

## Interface and behavior

- `ATTRES_KERNEL_MODE=fused` preserves the current production defaults.
- `ATTRES_KERNEL_MODE=reference` selects the unoptimized PyTorch AttnRes path.
- Any other value fails before distributed startup.
- The selected value appears in the launcher contract and node logs.
- Baseline and DAR behavior is unchanged.

The four-node launcher will pass the mode through the SSH boundary. The AttnRes
model launcher will translate `reference` into explicit zero values for all
relevant implementation flags before Python imports `residual_blocks.py`.

## Experiment

Run `fused` and `reference` for 10 steps on the same H100 hardware with identical
model configuration, initial seed, token files, sample order, micro/global batch
sizes, optimizer, and learning-rate schedule. Use separate empty output and log
directories. Capture per-step loss and, when practical, selected gradient and
parameter checksums.

Interpretation:

- Matching curves within normal BF16 numerical drift rules out the kernels as
  the source of the large long-run loss gap.
- A material, consistently growing difference requires bisection of individual
  optimized paths.
- A crash or OOM in `reference` is reported as an inconclusive comparison, not
  as evidence that the fused implementation is correct.

## Testing

- Launcher tests verify valid-mode propagation and logging.
- Launcher tests reject invalid modes before SSH startup.
- A shell dry run verifies that `reference` exports all required zero-valued
  flags and `fused` preserves current defaults.
- Existing launcher and residual-block tests must remain passing.

