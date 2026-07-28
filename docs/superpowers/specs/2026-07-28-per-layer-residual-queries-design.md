# Per-Layer Residual Queries Design

## Goal

Correct the AttnRes and Dual Attention Residual block architectures so every
atomic Transformer sublayer has its own residual query parameters instead of
sharing one attention query and one MLP query across an entire block.

For block size three:

- AttnRes owns three independent attention queries and three independent MLP
  queries: six queries total.
- Dual owns an independent two-stream query pair for each attention and MLP
  sublayer: six pairs, or twelve query vectors total.

## Selected representation

Each block stores two `nn.ParameterList` instances indexed by local atomic-layer
position.

- AttnRes: `attn_qs[i]` and `mlp_qs[i]`, each shaped `[hidden_size]`.
- Dual: `attn_qs[i]` and `mlp_qs[i]`, each shaped `[2, hidden_size]`.

Separate parameter objects make the architecture visible in state dictionaries,
gradient diagnostics, and distributed-optimizer parameter enumeration. A single
stacked tensor was rejected because it obscures per-layer ownership. Attaching
queries to `MCoreAtomicLayer` was rejected because that generic wrapper should
not own residual-architecture-specific state.

## Initialization and distributed behavior

Every AttnRes query is initialized to zero, preserving the existing per-query
initial condition. Every Dual query pair uses `init_dual_stream_query_`.
The preferred Dual stream is selected using the atomic layer's global layer
index, not only the enclosing block's start index.

Every query parameter retains the existing `sequence_parallel` attribute so
Megatron-Core synchronizes replicated parameters consistently across tensor
parallel ranks.

## Forward data flow

Both block forward loops enumerate local atomic layers. At layer index `i`, the
attention residual read uses `attn_qs[i]` and the MLP residual read uses
`mlp_qs[i]`.

History keys and values remain block-level because they do not depend on the
query. Query-dependent prefix and logit results must be cached by local layer
index and sublayer kind. No prefix or logits computed for one atomic layer may
be reused by another layer.

Paired optimized paths remain enabled: for a given atomic layer, the paired
kernel receives that layer's attention and MLP queries together. It computes a
new pair for every atomic layer.

All other Dual block parameters, including alpha, beta, residual gates, and
dynamic mixing matrices, remain block-scoped and unchanged.

## Checkpoints

Old checkpoints are intentionally incompatible. There is no shared-query
broadcast migration and no optimizer-state migration. Training after this
change starts from a fresh initialization.

## Validation

Tests must establish all of the following:

- Parameter counts and shapes match the block size for both architectures.
- Query objects and storage are distinct between atomic layers.
- Each sublayer receives its matching query during forward execution.
- Query-dependent history caches do not cross layer boundaries.
- Every per-layer query receives a finite, nonzero gradient in a representative
  backward pass.
- Query parameters retain sequence-parallel synchronization metadata.
- Existing fused/reference numerical tests and the complete repository test
  suite remain passing.

