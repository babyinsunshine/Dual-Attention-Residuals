# AttnRes Kernel A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable fused/reference AttnRes execution switch and use it for an identical 10-step H100 training comparison.

**Architecture:** A focused shell helper owns the mapping from `ATTRES_KERNEL_MODE` to the existing low-level AttnRes environment flags. Single-node and multi-node AttnRes launchers source that helper before Python starts; the four-node SSH controller validates, logs, and forwards the high-level mode to every node.

**Tech Stack:** Bash, Python `unittest`, PyTorch/Megatron-Core, Triton, SSH launcher tests.

## Global Constraints

- `fused` preserves the current production AttnRes defaults.
- `reference` disables all AttnRes Triton, fused epilogue, cached-prefix, and recomputation optimizations.
- Invalid values fail before distributed startup.
- Baseline and DAR behavior remains unchanged.
- The H100 jobs use identical model, seed, data, batch sizes, optimizer, and learning-rate schedule and run for exactly 10 steps.
- Preserve unrelated user changes in `dual_attentions_residuals/mcore_compat.py` and untracked benchmark configs.

---

### Task 1: AttnRes Mode Mapping

**Files:**
- Create: `scripts/attnres_kernel_mode.sh`
- Modify: `scripts/train_attnres_single_node.sh`
- Modify: `scripts/train_attnres_multi_node.sh`
- Create: `tests/test_attnres_kernel_mode.py`

**Interfaces:**
- Consumes: `ATTRES_KERNEL_MODE`, defaulting to `fused`.
- Produces: `configure_attnres_kernel_mode`, which exports the existing `MEGATRON_ATTRES_*` and `MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE` boolean variables.

- [ ] **Step 1: Write failing mode tests**

Create tests that source `scripts/attnres_kernel_mode.sh`, invoke
`configure_attnres_kernel_mode`, and print the relevant environment variables.
Assert that `reference` maps every variable below to `0`, `fused` maps them to
the listed current defaults, and an invalid value exits nonzero before invoking
Python.

```python
FLAGS = {
    "MEGATRON_ATTRES_TRITON_WEIGHTED_SUM": "1",
    "MEGATRON_ATTRES_CACHE_HISTORY_LOGITS_PAIR": "0",
    "MEGATRON_ATTRES_CACHE_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_ATTRES_RECOMPUTE_HISTORY_VALUES": "1",
    "MEGATRON_ATTRES_RECOMPUTE_HISTORY_KEYS": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_PREFIX_PAIR": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_KEY_NORM": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK": "1",
    "MEGATRON_ATTRES_TRITON_HISTORY_APPEND_PACK_STRICT": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE": "1",
    "MEGATRON_ATTRES_TRITON_CURRENT_FINALIZE_STRICT": "1",
    "MEGATRON_RESIDUAL_TE_ATTENTION_EPILOGUE": "1",
}
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python -m unittest tests.test_attnres_kernel_mode -v
```

Expected: failure because `scripts/attnres_kernel_mode.sh` does not exist.

- [ ] **Step 3: Implement the mode helper and source it from both AttnRes launchers**

Implement this public shape:

```bash
configure_attnres_kernel_mode() {
  local mode=${ATTRES_KERNEL_MODE:-fused}
  case "$mode" in
    fused)    # export the current production values ;;
    reference) # export 0 for every listed value ;;
    *) echo "ATTRES_KERNEL_MODE must be fused or reference, got: $mode" >&2; return 2 ;;
  esac
  export ATTRES_KERNEL_MODE="$mode"
}
```

Source and call the helper before constructing the Python command in both
AttnRes launchers. Print `ATTRES_KERNEL_MODE=<value>` to the node log.

- [ ] **Step 4: Verify mode tests pass**

Run:

```bash
python -m unittest tests.test_attnres_kernel_mode -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the mode mapping**

```bash
git add scripts/attnres_kernel_mode.sh scripts/train_attnres_single_node.sh \
  scripts/train_attnres_multi_node.sh tests/test_attnres_kernel_mode.py
git commit -m "feat: add AttnRes kernel execution modes"
```

### Task 2: Four-Node Mode Propagation

**Files:**
- Modify: `scripts/train_7b_attnres_4node.sh`
- Modify: `scripts/launch_7b_ssh_common.sh`
- Modify: `tests/test_three_cluster_7b_launchers.py`

**Interfaces:**
- Consumes: validated `ATTRES_KERNEL_MODE=fused|reference` from the controller environment.
- Produces: the same literal value in every remote `setsid env` invocation.

- [ ] **Step 1: Write failing propagation tests**

Extend the existing fake-SSH launcher tests with these assertions:

```python
self.assertIn("ATTRES_KERNEL_MODE=fused", completed.stdout)
env["ATTRES_KERNEL_MODE"] = "reference"
self.assertIn("ATTRES_KERNEL_MODE=reference", completed.stdout)
```

Inspect fake SSH call records for each rank and assert that the remote launch
command receives `reference`. Add an invalid-mode case expecting:

```text
ATTRES_KERNEL_MODE must be fused or reference
```

and no SSH calls.

- [ ] **Step 2: Verify propagation tests fail**

Run the exact new test methods with:

```bash
python -m unittest \
  tests.test_three_cluster_7b_launchers.ThreeCluster7BLauncherTests.test_attnres_kernel_mode_contract \
  tests.test_three_cluster_7b_launchers.ThreeCluster7BLauncherTests.test_attnres_kernel_mode_rejects_invalid_value -v
```

Expected: assertions fail because the common launcher neither prints nor
forwards the mode.

- [ ] **Step 3: Implement validation, logging, and SSH forwarding**

Default the AttnRes entry point to `fused`. In the common controller, validate
that the value is `fused` or `reference` when `MODEL_NAME=attnres`, print it in
`print_contract`, append it to the remote launch arguments, and place it in the
`setsid env` environment as:

```bash
ATTRES_KERNEL_MODE="$attres_kernel_mode" \
```

Use `fused` as the harmless value for non-AttnRes entry points without printing
an AttnRes contract line.

- [ ] **Step 4: Run launcher tests**

Run:

```bash
python -m unittest tests.test_three_cluster_7b_launchers -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit propagation**

```bash
git add scripts/train_7b_attnres_4node.sh scripts/launch_7b_ssh_common.sh \
  tests/test_three_cluster_7b_launchers.py
git commit -m "feat: propagate AttnRes kernel mode across nodes"
```

### Task 3: Verification and H100 A/B

**Files:**
- Verify only: repository test suite and H100 run logs.

**Interfaces:**
- Consumes: `ATTRES_KERNEL_MODE=fused|reference` and two empty output directories.
- Produces: paired 10-step logs with per-step losses and a numerical comparison.

- [ ] **Step 1: Run static and focused verification**

Run:

```bash
bash -n scripts/attnres_kernel_mode.sh scripts/train_attnres_single_node.sh \
  scripts/train_attnres_multi_node.sh scripts/train_7b_attnres_4node.sh \
  scripts/launch_7b_ssh_common.sh
python -m unittest tests.test_attnres_kernel_mode \
  tests.test_three_cluster_7b_launchers -v
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Run fused training for 10 steps on H100**

Launch with a fresh output directory, `TRAIN_ITERS=10`, and
`ATTRES_KERNEL_MODE=fused`. Record the exact command, GPU model, git commit,
environment versions, seed, token files, and per-step loss.

- [ ] **Step 3: Run reference training for 10 steps on the same H100**

Repeat from the same initialization and sample stream with a different empty
output directory and `ATTRES_KERNEL_MODE=reference`. Do not resume either run.

- [ ] **Step 4: Compare logs**

Extract iteration and loss from both logs and report absolute per-step
differences. If the difference is only BF16-scale and does not grow materially,
rule out the kernel as the cause of the observed `0.2` long-run gap. Otherwise,
identify the first divergent step for path-level bisection.

- [ ] **Step 5: Commit any test-only correction, if required**

Only if verification exposes a defect in the mode switch, fix it using a new
failing regression test and commit the focused correction separately. Do not
modify unrelated user files.

