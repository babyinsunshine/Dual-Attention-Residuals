# Per-Sublayer Residual State Design

## Goal

Correct the AttnRes and Dual Attention Residual block architectures so every
atomic Transformer sublayer has its own residual state instead of sharing that
state across an entire block.

For block size three:

- AttnRes owns three independent attention queries and three independent MLP
  queries: six queries total.
- Dual owns an independent two-stream query pair for each attention and MLP
  sublayer: six pairs, or twelve query vectors total.
- Every Dual attention and MLP sublayer also owns independent static alpha and
  beta parameters, dynamic alpha/beta and residual-mixing matrices, residual
  gates/scales, mixing norm, and history-key norm.

## Selected representation

AttnRes stores two `nn.ParameterList` instances indexed by local atomic-layer
position.

- AttnRes: `attn_qs[i]` and `mlp_qs[i]`, each shaped `[hidden_size]`.
- Dual stores `attn_states[i]` and `mlp_states[i]` in `nn.ModuleList`
  containers. Each state owns its `[2, hidden_size]` query and the complete set
  of mixing parameters and norms used by that residual read.

Separate state modules make ownership visible in state dictionaries, gradient
diagnostics, and distributed-optimizer parameter enumeration. A single stacked
tensor was rejected because it obscures per-sublayer ownership and permits the
query to be indexed without indexing its matching gates and matrices. Attaching
state to `MCoreAtomicLayer` was rejected because that generic wrapper should not
own residual-architecture-specific state.

## Initialization and distributed behavior

Every AttnRes query is initialized to zero, preserving the existing per-query
initial condition. Every Dual query pair uses `init_dual_stream_query_`.
The preferred Dual stream is selected using the atomic layer's global layer
index, not only the enclosing block's start index.

Every learnable residual-state parameter retains the existing
`sequence_parallel` attribute so Megatron-Core synchronizes replicated
parameters consistently across tensor parallel ranks.

## Forward data flow

Both block forward loops enumerate local atomic layers. At layer index `i`, the
Dual attention residual read uses the complete `attn_states[i]` object and the
MLP residual read uses `mlp_states[i]`. No parameter from one state may be used
for another atomic sublayer.

History values remain block-level. History keys use the selected sublayer's
history-key norm. Because `FixedRMSNorm` is parameterless, numerically identical
norm instances may reuse a cached key tensor as a common-subexpression
optimization, keyed by norm type and configuration; norms with parameters or
different configuration may not share that result. Query-dependent prefix and
logit results remain cached by local layer index and sublayer kind.

Paired optimized paths remain enabled: for a given atomic layer, the paired
kernel receives that layer's attention and MLP queries together. It computes a
new pair for every atomic layer.

The output aggregation state remains model-output scoped. Only the structural
history list and derived caches are shared by the enclosing block.

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
- Every per-layer query receives a finite gradient and fires its grad-ready hook
  in a representative backward pass. A singleton retrieval is mathematically
  query-independent, so its retained dependency must produce an exact-zero
  gradient rather than a missing gradient.
- Query parameters retain sequence-parallel synchronization metadata.
- All Dual residual-state parameter and norm objects are distinct across the
  six atomic sublayers in a three-layer block.
- Forward wiring supplies each residual read with exactly its matching complete
  state object, and every learnable state parameter receives a finite gradient.
- Existing fused/reference numerical tests and the complete repository test
  suite remain passing.
