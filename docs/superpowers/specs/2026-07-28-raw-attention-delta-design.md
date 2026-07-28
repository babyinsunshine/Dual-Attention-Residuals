# Raw Attention Delta Fix Design

## Problem

The Megatron adapter currently reconstructs an attention branch delta as
`BF16(BF16(hidden + raw_attention) - hidden)`.  This is algebraically equal to
`raw_attention` in exact arithmetic, but loses small updates after the first
BF16 rounding.  It differs from the DeepSpeed reference, which consumes the
raw attention branch output directly.  The same reconstruction is embedded in
the DAR fused attention epilogue and corrupts the gradient of its beta routing
coefficients.

## Design

`MCoreAtomicLayer.attention_delta()` will delegate to
`attention_raw_output()`.  The residual-block helper will use that same method
regardless of whether the optional TE epilogue optimization is enabled, so the
fallback and optimized modes have identical semantics.

The DAR fused epilogue will use `raw_attention` directly as its delta, merge it
with the two residual streams, and save it for the beta-gradient calculation.
The public Python function signature remains stable to minimize caller churn;
the attention-input tensor is retained as a validated shape/dtype input until a
separate API cleanup.

## Compatibility

No parameters, state-dict keys, model shapes, configuration keys, or launcher
interfaces change. Existing checkpoints remain loadable. Resuming an old run
is mechanically supported, but a fresh run is required for a scientifically
fair comparison because earlier updates used the lossy semantics.

## Verification

Tests will assert that the adapter returns raw branch output without a residual
round trip, and CUDA tests will compare both attention-delta kernels against a
direct-raw PyTorch reference in BF16 for forward and backward. The adversarial
fixture uses a large residual and a small nonzero branch output so the old code
fails deterministically. The complete CPU test suite and the H800 CUDA tests
must pass before integration.
