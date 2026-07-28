# Raw Attention Delta Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AttnRes and DAR consume the raw BF16 attention branch output without a lossy residual add/subtract round trip.

**Architecture:** Keep the existing adapter and Triton public interfaces, but change their delta semantics to direct raw attention. Protect both the ordinary adapter path and the fused DAR path with deterministic BF16 regression tests.

**Tech Stack:** Python 3.10, PyTorch, Megatron-Core 0.16.1, Triton, pytest, NVIDIA H800.

## Global Constraints

- Do not change model parameters, state-dict keys, tensor shapes, configuration keys, or launcher interfaces.
- Preserve BF16 forward and backward behavior except for removing the lossy attention residual reconstruction.
- Existing checkpoints must remain loadable.
- Run CUDA-specific tests on the supplied single H800.

---

### Task 1: Lock the raw-attention adapter contract

**Files:**
- Create: `tests/test_attention_delta_semantics.py`
- Modify: `dual_attentions_residuals/model.py:335-349`

**Interfaces:**
- Consumes: `MCoreAtomicLayer.attention_raw_output(hidden_states, attention_mask=None, rotary_pos_emb=None)`.
- Produces: `MCoreAtomicLayer.attention_delta(...) -> torch.Tensor` with direct raw-branch semantics.

- [ ] **Step 1: Write the failing adapter test**

Construct an `MCoreAtomicLayer` around a small real `nn.Module` test double whose `_forward_attention` returns a BF16 residual-added value while `self_attention` returns a hand-chosen nonzero raw branch value. Assert that `attention_delta()` equals the literal raw value exactly.

- [ ] **Step 2: Run the adapter test and verify RED**

Run: `python -m pytest tests/test_attention_delta_semantics.py::test_attention_delta_returns_raw_branch_without_bf16_residual_round_trip -q`

Expected: FAIL because the current implementation returns zero for at least one nonzero raw BF16 element.

- [ ] **Step 3: Implement the minimal adapter change**

Replace the `_forward_attention(...)-hidden_states` reconstruction in `attention_delta()` with delegation to `attention_raw_output()` using the same mask and rotary arguments.

- [ ] **Step 4: Run the adapter test and verify GREEN**

Run: `python -m pytest tests/test_attention_delta_semantics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the adapter change**

```bash
git add tests/test_attention_delta_semantics.py dual_attentions_residuals/model.py
git commit -m "fix: use raw attention output for residual delta"
```

### Task 2: Correct fused BF16 attention delta and DAR beta gradients

**Files:**
- Modify: `tests/test_attention_delta_semantics.py`
- Modify: `dual_attentions_residuals/dual_residual_triton.py:85-137`

**Interfaces:**
- Consumes: `triton_te_attention_delta(attention_input, raw_attention)` and `triton_dual_te_attention_epilogue(attention_input, raw_attention, residual0, residual1, beta, has_residual=...)`.
- Produces: exact raw-attention delta semantics and beta gradients derived from raw attention.

- [ ] **Step 1: Write failing CUDA tests**

Use BF16 `attention_input=16.0` and `raw_attention=0.03125`, for which the old residual round trip becomes zero. Assert that `triton_te_attention_delta()` returns the raw value exactly. For the dual epilogue, assert forward output and gradients for `raw_attention`, residual streams, and `beta` equal a direct PyTorch expression using `raw_attention`.

- [ ] **Step 2: Run the CUDA tests on H800 and verify RED**

Run remotely: `python -m pytest tests/test_attention_delta_semantics.py -q`

Expected: the adapter test passes after Task 1, while the two Triton tests fail on delta output and DAR beta gradient.

- [ ] **Step 3: Implement the minimal Triton change**

In both `_te_attention_delta_kernel` and `_dual_te_attention_epilogue_kernel`, assign `delta` directly from the loaded raw attention value. Keep existing output types, merged-output logic, saved tensor, and backward interface unchanged; the saved delta now contains the correct raw value.

- [ ] **Step 4: Run the CUDA tests and verify GREEN**

Run remotely: `python -m pytest tests/test_attention_delta_semantics.py -q`

Expected: PASS with exact direct-raw forward results and matching gradients.

- [ ] **Step 5: Commit the fused-kernel change**

```bash
git add tests/test_attention_delta_semantics.py dual_attentions_residuals/dual_residual_triton.py
git commit -m "fix: preserve raw BF16 attention deltas in fused paths"
```

### Task 3: Full regression and integration

**Files:**
- Verify only; no additional production files.

**Interfaces:**
- Consumes: the adapter and fused-kernel changes from Tasks 1 and 2.
- Produces: a tested branch ready to merge into `main` without touching unrelated user changes.

- [ ] **Step 1: Run the complete local CPU suite**

Run: `python -m pytest -q`

Expected: exit code 0.

- [ ] **Step 2: Run the focused H800 CUDA suite again from a clean copied tree**

Run remotely: `python -m pytest tests/test_attention_delta_semantics.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Inspect the final diff and status**

Run: `git diff main...HEAD --check && git status --short`

Expected: no whitespace errors and no uncommitted files.

- [ ] **Step 4: Integrate the branch locally**

Merge `fix/raw-attention-delta` into `main` without resetting or overwriting the pre-existing modifications in the main worktree, then rerun the focused CPU test from `main`.
