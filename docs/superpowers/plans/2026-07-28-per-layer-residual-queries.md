# Per-Layer Residual Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every AttnRes and Dual atomic attention/MLP sublayer its own residual query parameters and prevent query-dependent cache reuse across layers.

**Architecture:** Each residual block owns `attn_qs` and `mlp_qs` `nn.ParameterList` objects indexed by local atomic-layer position. Forward loops select the matching pair at every layer, while history values remain block-scoped and query-dependent prefix/logit caches become layer-scoped.

**Tech Stack:** Python, PyTorch `nn.ParameterList`, Megatron-Core parameter metadata, Triton paired-prefix interfaces, `pytest`/`unittest`.

## Global Constraints

- For block size three, AttnRes has six independent query vectors.
- For block size three, Dual has six independent two-stream query pairs, or twelve query vectors.
- AttnRes queries keep zero initialization.
- Dual query pairs use `init_dual_stream_query_` with the atomic layer's global layer index.
- All query parameters retain `sequence_parallel=True` metadata.
- Alpha, beta, residual gates, key projections, norms, and dynamic mixing matrices remain block-scoped.
- Existing paired Triton paths remain enabled but execute once per atomic layer.
- Old checkpoints are intentionally incompatible and receive no migration path.
- Preserve unrelated user changes in `dual_attentions_residuals/mcore_compat.py` and untracked benchmark configs.

---

### Task 1: Lock the Parameter Architecture with Failing Tests

**Files:**
- Create: `tests/test_per_layer_residual_queries.py`
- Inspect: `dual_attentions_residuals/residual_blocks.py`

**Interfaces:**
- Consumes: `AttentionResidualBlock(config, layers)` and `DualAttentionResidualBlock(config, layers, layer_index)`.
- Produces: regression tests defining `attn_qs` and `mlp_qs` parameter counts, shapes, initialization, identity, names, and metadata.

- [ ] **Step 1: Add minimal test fixtures**

Create a `SimpleNamespace` configuration fixture containing the model/train
fields required by both constructors and a three-element list of fake atomic
layers. Use hidden size eight and block size three.

- [ ] **Step 2: Add failing AttnRes structure assertions**

Assert:

```python
assert len(block.attn_qs) == 3
assert len(block.mlp_qs) == 3
assert all(tuple(q.shape) == (8,) for q in block.attn_qs)
assert all(tuple(q.shape) == (8,) for q in block.mlp_qs)
assert len({q.data_ptr() for q in [*block.attn_qs, *block.mlp_qs]}) == 6
assert all(torch.count_nonzero(q) == 0 for q in [*block.attn_qs, *block.mlp_qs])
assert all(q.sequence_parallel for q in [*block.attn_qs, *block.mlp_qs])
```

Also assert that state-dict keys include `attn_qs.0` through `attn_qs.2`
and `mlp_qs.0` through `mlp_qs.2`, with no legacy
`attn_res_proj.weight` or `mlp_res_proj.weight` keys.

- [ ] **Step 3: Add failing Dual structure assertions**

Assert three `[2, 8]` parameters in each list, twelve distinct row storage
locations, sequence-parallel metadata on every parameter, and state-dict names
`attn_qs.0` through `mlp_qs.2`. Patch `init_dual_stream_query_` and assert its
preferred-stream calls follow global layers `5, 6, 7` as `1, 0, 1` when the
block begins at `layer_index=5`.

- [ ] **Step 4: Verify the tests fail for the architectural reason**

Run:

```bash
python -m pytest tests/test_per_layer_residual_queries.py -q
```

Expected: failures report missing `attn_qs`/`mlp_qs` because the implementation
still exposes one shared query per block.

### Task 2: Implement Independent Query Parameters

**Files:**
- Modify: `dual_attentions_residuals/residual_blocks.py:1135-1171`
- Modify: `dual_attentions_residuals/residual_blocks.py:1283-1302`
- Modify: `dual_attentions_residuals/residual_blocks.py:1554-1645`
- Test: `tests/test_per_layer_residual_queries.py`

**Interfaces:**
- Consumes: block-local atomic-layer count and global block start layer.
- Produces: `AttentionResidualBlock.attn_qs`, `AttentionResidualBlock.mlp_qs`, `DualAttentionResidualBlock.attn_qs`, and `DualAttentionResidualBlock.mlp_qs`.

- [ ] **Step 1: Replace AttnRes shared projections**

Create two `nn.ParameterList` objects. Each item is a distinct zero-initialized
`nn.Parameter` shaped `[hidden_size]` with `_train_param_dtype(config)`. Pass the
query tensor directly into `_block_attn_res`; remove the internal
`proj.weight.squeeze(0)` lookup.

- [ ] **Step 2: Replace Dual shared query pairs**

Create two `nn.ParameterList` objects with one `[2, hidden_size]` parameter per
atomic layer. Initialize each pair independently using:

```python
preferred_stream = (layer_index + local_layer_index) % self.num_streams
init_dual_stream_query_(query, preferred_stream=preferred_stream, balanced=_DUAL_Q_BALANCED)
```

- [ ] **Step 3: Mark every new query parameter for sequence parallelism**

Flatten both query lists into the existing `_mark_sequence_parallel_parameters`
call. Do not change metadata on any other parameter.

- [ ] **Step 4: Run the structure tests**

Run:

```bash
python -m pytest tests/test_per_layer_residual_queries.py -q
```

Expected: structure and initialization tests pass; forward-selection tests added
in Task 3 may still fail.

### Task 3: Use the Matching Query and Isolate Caches per Layer

**Files:**
- Modify: `dual_attentions_residuals/residual_blocks.py:1364-1506`
- Modify: `dual_attentions_residuals/residual_blocks.py:2231-2555`
- Modify: `tests/test_per_layer_residual_queries.py`

**Interfaces:**
- Consumes: the four per-layer `ParameterList` objects from Task 2.
- Produces: layer-indexed query selection and `(layer_index, sublayer_kind)` cache ownership.

- [ ] **Step 1: Add failing forward-selection tests**

Patch the block residual-read methods with recording fakes, disable GPU-only
optimized branches, and run a three-layer CPU forward/backward pass. Assert the
observed query order is:

```text
attn_qs[0], mlp_qs[0], attn_qs[1], mlp_qs[1], attn_qs[2], mlp_qs[2]
```

Assert every query has a finite, nonzero gradient.

- [ ] **Step 2: Add failing paired-prefix cache tests**

Patch `_attres_history_prefix_pair` and `_dual_history_prefix_pair` with
recording fakes. With paired-prefix caching enabled, assert exactly three calls
per three-layer block and assert call `i` receives `attn_qs[i]` and
`mlp_qs[i]`. This prevents a first-layer prefix from being reused by later
layers.

- [ ] **Step 3: Enumerate AttnRes layers and select matching queries**

Change the loop to `for layer_index, layer in enumerate(self.layers)`. Pass
`attn_qs[layer_index]` and `mlp_qs[layer_index]` into residual reads. Store
paired prefix/logit results in dictionaries keyed by local layer index. Clear
the dictionaries whenever history keys or values change.

- [ ] **Step 4: Enumerate Dual layers and select matching query pairs**

Select `attn_qs[layer_index]` and `mlp_qs[layer_index]`. Key prefix and logit
caches by `(layer_index, "attn")` and `(layer_index, "mlp")`. The paired path
must compute and store both entries for the same layer only. Clear caches on
every history append.

- [ ] **Step 5: Run all per-layer query tests**

Run:

```bash
python -m pytest tests/test_per_layer_residual_queries.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the architecture fix**

```bash
git add dual_attentions_residuals/residual_blocks.py tests/test_per_layer_residual_queries.py
git commit -m "fix: give each residual sublayer an independent query"
```

### Task 4: Full Verification and GPU Smoke Test

**Files:**
- Verify: `dual_attentions_residuals/residual_blocks.py`
- Verify: `tests/test_per_layer_residual_queries.py`
- Verify: existing repository tests.

**Interfaces:**
- Consumes: corrected per-layer query architecture.
- Produces: evidence that CPU tests, fused/reference paths, parameter gradients, and training startup remain valid.

- [ ] **Step 1: Run static and full repository checks**

Run:

```bash
python -m pytest tests/test_per_layer_residual_queries.py -q
python -m unittest discover -s tests
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Run a CUDA block smoke test**

On the available H800, instantiate three-layer AttnRes and Dual blocks, execute
forward/backward, and print every query's gradient norm. Assert six nonzero
AttnRes query gradients and six nonzero Dual pair gradients.

- [ ] **Step 3: Inspect parameter names and counts**

Print matching state-dict keys for block size three and verify six AttnRes query
entries and six Dual query-pair entries. Confirm there are no legacy block-shared
query names.

- [ ] **Step 4: Merge locally to main and rerun the full suite**

Fast-forward the verified feature branch into `main`, preserve unrelated dirty
files, and run `python -m unittest discover -s tests` again on the merged tree.

