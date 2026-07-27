# Three 7B Four-Node Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide safe background launch commands for Baseline, AttnRes, and DAR 7B training on three independent four-node/eight-GPU groups for 40,000 iterations at global batch size 128.

**Architecture:** Three thin model entry scripts bind model-specific config, hostfile, output, log, and PID defaults. A shared Bash controller parses and validates a four-host file, performs all-node SSH preflight, reconstructs the CUDA/PyTorch library path remotely, launches one `torch.distributed.run` job per node, and manages background start/status/logs/stop lifecycle. Existing model-specific multi-node scripts remain the final torchrun boundary and are extended to forward the complete schedule override contract.

**Tech Stack:** Bash 4+, OpenSSH, PyTorch `torch.distributed.run`, Python `unittest`, fake-SSH integration tests.

## Global Constraints

- Shared server project root is `/gemini/space/internship/yuxingda/dual_attentions_residuals_megatron`.
- Python interpreter is `<project>/.venv-train/bin/python` on every host.
- Each hostfile contains exactly four unique non-comment hosts; its first host is rank 0 and `MASTER_ADDR`.
- Baseline uses `hostfile_baseline`, AttnRes uses `hostfile_attnres`, and DAR uses `hostfile_dar` at repository root.
- Each experiment uses 4 nodes × 8 GPUs, TP/PP/EP/CP `2/1/4/1`, DP 16, micro/global batch `4/128`, and gradient accumulation 2.
- Each experiment uses 40,000 iterations, 2,000 warmup iterations, cosine decay through iteration 40,000, peak/minimum LR `3e-4/3e-5`, save/eval interval 10,000, and 20 eval iterations.
- Use learned MoE routing with `MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0`.
- Only finalized `data/fineweb_100BT_pretok/train.bin` and `val.bin` are accepted; never use `.partial` files.
- Do not add resume behavior; a non-empty output checkpoint directory must be rejected.
- Preserve unrelated user changes in `dual_attentions_residuals/mcore_compat.py` and untracked 1B configuration files.

---

## File structure

- Modify `scripts/train_baseline_multi_node.sh`: forward schedule/batch/eval environment overrides to `train.py`.
- Modify `scripts/train_attnres_multi_node.sh`: same forwarding for AttnRes.
- Modify `scripts/train_dual_multi_node.sh`: same forwarding for DAR.
- Create `scripts/launch_7b_ssh_common.sh`: hostfile parsing, validation, remote preflight/launch, CUDA environment, logs, and lifecycle operations.
- Create `scripts/train_7b_baseline_4node.sh`: Baseline defaults and delegation to the common controller.
- Create `scripts/train_7b_attnres_4node.sh`: AttnRes defaults and delegation to the common controller.
- Create `scripts/train_7b_dar_4node.sh`: DAR defaults and delegation to the common controller.
- Create `tests/test_7b_multi_node_overrides.py`: behavioral tests for final torchrun argument forwarding.
- Create `tests/test_three_cluster_7b_launchers.py`: real-shell/fake-SSH tests for host parsing, model selection, preflight, rank launch, and lifecycle.
- Modify `README.md`: exact hostfile formats and start/status/logs/stop commands.

---

### Task 1: Complete the model-specific multi-node override contract

**Files:**
- Modify: `scripts/train_baseline_multi_node.sh`
- Modify: `scripts/train_attnres_multi_node.sh`
- Modify: `scripts/train_dual_multi_node.sh`
- Create: `tests/test_7b_multi_node_overrides.py`

**Interfaces:**
- Consumes: required `NNODES`, `NODE_RANK`, `MASTER_ADDR`; optional `MASTER_PORT`, `GPUS_PER_NODE`, `PYTHON_BIN`, and training environment variables.
- Produces: model-specific `torch.distributed.run` command containing exact `train.py` CLI overrides.

- [ ] **Step 1: Write a failing table-driven test for all three launchers**

Create `tests/test_7b_multi_node_overrides.py`. For each launcher, run the real
shell script with `DRY_RUN=1` and these literal environment values:

```python
EXPECTED = {
    "train_baseline_multi_node.sh": "configs/7b_moe_baseline.yaml",
    "train_attnres_multi_node.sh": "configs/7b_moe_attnres.yaml",
    "train_dual_multi_node.sh": "configs/7b_moe_dual.yaml",
}

env.update({
    "NNODES": "4",
    "NODE_RANK": "2",
    "MASTER_ADDR": "10.0.0.1",
    "MASTER_PORT": "29600",
    "GPUS_PER_NODE": "8",
    "PYTHON_BIN": "/opt/train/python",
    "TRAIN_TOKENS_FILE": "/data/train.bin",
    "VAL_TOKENS_FILE": "/data/val.bin",
    "OUTPUT_DIR": "/output/model",
    "TRAIN_ITERS": "40000",
    "MICRO_BATCH_SIZE": "4",
    "GLOBAL_BATCH_SIZE": "128",
    "SAVE_INTERVAL": "10000",
    "EVAL_INTERVAL": "10000",
    "EVAL_ITERS": "20",
    "DRY_RUN": "1",
})
```

Assert return code zero and literal command arguments for config, ranks, token
files, output, train iterations, both batch sizes, save interval, eval interval,
and eval iterations. The expected values are literals rather than values
derived through launcher helpers.

- [ ] **Step 2: Run the test and confirm RED**

Run:

```bash
python -m unittest tests.test_7b_multi_node_overrides -v
```

Expected: all three cases fail because the current scripts omit
`--micro-batch-size`, `--global-batch-size`, `--save-interval`,
`--eval-interval`, and `--eval-iters`.

- [ ] **Step 3: Add the minimal forwarding logic to every multi-node script**

After the existing `TRAIN_ITERS` append, add the same five guarded appends:

```bash
[[ -n ${MICRO_BATCH_SIZE:-} ]] && args+=(--micro-batch-size "$MICRO_BATCH_SIZE")
[[ -n ${GLOBAL_BATCH_SIZE:-} ]] && args+=(--global-batch-size "$GLOBAL_BATCH_SIZE")
[[ -n ${SAVE_INTERVAL:-} ]] && args+=(--save-interval "$SAVE_INTERVAL")
[[ -n ${EVAL_INTERVAL:-} ]] && args+=(--eval-interval "$EVAL_INTERVAL")
[[ -n ${EVAL_ITERS:-} ]] && args+=(--eval-iters "$EVAL_ITERS")
```

Do not change config selection or torchrun construction.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run:

```bash
python -m unittest tests.test_7b_multi_node_overrides -v
bash -n scripts/train_baseline_multi_node.sh \
  scripts/train_attnres_multi_node.sh \
  scripts/train_dual_multi_node.sh
```

Expected: tests pass and `bash -n` exits zero.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/train_baseline_multi_node.sh \
  scripts/train_attnres_multi_node.sh \
  scripts/train_dual_multi_node.sh \
  tests/test_7b_multi_node_overrides.py
git commit -m "feat: forward 7b multinode schedule overrides"
```

---

### Task 2: Add the common SSH controller and three background entry scripts

**Files:**
- Create: `scripts/launch_7b_ssh_common.sh`
- Create: `scripts/train_7b_baseline_4node.sh`
- Create: `scripts/train_7b_attnres_4node.sh`
- Create: `scripts/train_7b_dar_4node.sh`
- Create: `tests/test_three_cluster_7b_launchers.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: entry-script defaults `MODEL_NAME`, `HOSTFILE`, `MODEL_LAUNCHER`, `CONFIG_FILE`, `OUTPUT_DIR`, `LOG_DIR`, and `PID_FILE`; optional action `start|status|logs|stop`.
- Produces: a supervised background controller and four node log streams for exactly one model/hostfile pair.

- [ ] **Step 1: Write failing hostfile and model-selection tests**

Create a temporary runtime with executable fake Python, finalized train/val
files, four-host files, and fake SSH. Run each real entry script with
`DRY_RUN=1` and assert these literal mappings:

```python
EXPECTED = {
    "train_7b_baseline_4node.sh": (
        "hostfile_baseline", "configs/7b_moe_baseline.yaml", "baseline"
    ),
    "train_7b_attnres_4node.sh": (
        "hostfile_attnres", "configs/7b_moe_attnres.yaml", "attnres"
    ),
    "train_7b_dar_4node.sh": (
        "hostfile_dar", "configs/7b_moe_dual.yaml", "dar"
    ),
}
```

Assert the four rank mappings, first host as master, 40,000/4/128/2,000,
10,000/10,000/20 schedule, finalized token paths, learned-routing value zero,
and separate output/log/PID paths. Add rejection cases for three hosts, five
hosts, duplicate hosts, extra tokens on a line, missing final token files, and
non-empty output directories.

- [ ] **Step 2: Write failing fake-SSH controller tests**

The fake SSH consumes the remote script from stdin and records arguments. In
normal mode it returns zero for `preflight` and `launch`; in a configured
failure mode one preflight returns 23. Assert:

```python
self.assertEqual(preflight_count, 4)
self.assertEqual(launch_count, 4)
self.assertEqual(node_ranks, ["0", "1", "2", "3"])
self.assertEqual(master_addresses, ["10.0.0.1"] * 4)
```

For a failed preflight, assert return code nonzero and `launch_count == 0`.
Verify each node log is distinct and names the correct model/rank.

- [ ] **Step 3: Write failing background lifecycle tests**

Use a fake SSH whose launch mode blocks and exits on `TERM`. Invoke the entry
script with `start`, poll up to five seconds for `status` to report `RUNNING`,
then invoke `stop` and assert the PID no longer exists. Separately write PID
`99999999`, assert `status` reports `STOPPED`, and verify a subsequent start is
not rejected as already running.

- [ ] **Step 4: Run focused tests and confirm RED**

Run:

```bash
python -m unittest tests.test_three_cluster_7b_launchers -v
```

Expected: failure because the common controller and entry scripts do not yet
exist.

- [ ] **Step 5: Implement strict hostfile parsing and configuration validation**

In `scripts/launch_7b_ssh_common.sh`, require every entry-provided variable and
parse one host per line while removing comments and blank lines. Use an
associative array to reject duplicates and require exactly four hosts:

```bash
declare -a HOSTS=()
declare -A SEEN=()
while IFS= read -r raw || [[ -n $raw ]]; do
  line=${raw%%#*}
  read -r host extra <<< "$line"
  [[ -z ${host:-} ]] && continue
  [[ -z ${extra:-} ]] || die "hostfile line must contain one host: $raw"
  [[ -z ${SEEN[$host]+x} ]] || die "duplicate host: $host"
  SEEN[$host]=1
  HOSTS+=("$host")
done < "$HOSTFILE"
(( ${#HOSTS[@]} == 4 )) || die "hostfile must contain exactly 4 unique hosts"
MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}
```

Resolve relative paths against repository root. Validate positive integer
values, `GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * 16) == 0`, finalized filenames,
executable Python, readable config/data files, and an empty/nonexistent output
directory. Print the complete resolved contract before any SSH call. Under
`DRY_RUN=1`, stop after printing.

- [ ] **Step 6: Implement remote preflight and rank launch**

Embed a remote Bash program with two modes. `preflight` validates paths, exactly
eight visible GPUs, and imports:

```bash
"$python_bin" -c \
  'import torch; import megatron.core; import transformer_engine.pytorch; import triton'
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( gpu_count == 8 )) || { echo "expected 8 GPUs, found $gpu_count" >&2; exit 2; }
```

`launch` calculates site-packages, PyTorch libraries, NVIDIA wheel library
directories, and exports:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=$CUDA_HOME
export LD_LIBRARY_PATH="$clean_ld"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0
export MEGATRON_ENABLE_LR_SCHEDULER=1
export MEGATRON_LR_WARMUP_ITERS=2000
export MEGATRON_LR_DECAY_ITERS=40000
export MEGATRON_LR_DECAY_STYLE=cosine
export MEGATRON_LR_WARMUP_INIT_LR=0
```

Then `exec env` the selected existing model launcher with `NNODES=4`, rank,
master, eight GPUs, Python, final data paths, unique output, and the literal
schedule values. Run all preflights sequentially; only after all pass, launch
the four SSH sessions concurrently. Install `TERM/INT/EXIT` cleanup that
terminates active SSH child PIDs and waits for them. After all four SSH launch
children exist and remain alive through the initial poll, atomically publish
the ready marker consumed by `start`.

- [ ] **Step 7: Implement lifecycle operations and entry scripts**

The common script accepts `start` by default plus `status`, `logs`, `stop`, and
private `__controller` actions. `start` removes a stale PID, creates log
directories, starts `__controller` with `nohup`, writes its PID atomically, and
polls until a ready marker exists or the controller exits. `status` validates a
numeric PID with `kill -0`. `logs` executes `tail -n 100 -f` on the controller
log. `stop` sends `TERM`, waits up to 30 seconds, reports failure without using
`KILL`, and removes only its own PID/ready files after confirmed exit.

Each entry script contains literal defaults and delegates:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MODEL_NAME=baseline
export HOSTFILE=${HOSTFILE:-$ROOT/hostfile_baseline}
export MODEL_LAUNCHER=$ROOT/scripts/train_baseline_multi_node.sh
export CONFIG_FILE=$ROOT/configs/7b_moe_baseline.yaml
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/7b_moe_baseline_4node_40k}
export LOG_DIR=${LOG_DIR:-$ROOT/logs/7b_moe_baseline_4node_40k}
export PID_FILE=${PID_FILE:-$LOG_DIR/controller.pid}
exec "$ROOT/scripts/launch_7b_ssh_common.sh" "${1:-start}"
```

Repeat with `attnres/hostfile_attnres/train_attnres_multi_node.sh/
7b_moe_attnres.yaml` and `dar/hostfile_dar/train_dual_multi_node.sh/
7b_moe_dual.yaml`; do not derive these model mappings dynamically.

- [ ] **Step 8: Run focused tests and refine only observed failures**

Run:

```bash
python -m unittest tests.test_three_cluster_7b_launchers -v
bash -n scripts/launch_7b_ssh_common.sh \
  scripts/train_7b_baseline_4node.sh \
  scripts/train_7b_attnres_4node.sh \
  scripts/train_7b_dar_4node.sh
```

Expected: all focused tests pass and all scripts parse.

- [ ] **Step 9: Document exact production operations**

Add a README section containing this hostfile format:

```text
10.0.0.1
10.0.0.2
10.0.0.3
10.0.0.4
```

Document final-data prerequisite and the commands:

```bash
bash scripts/train_7b_baseline_4node.sh start
bash scripts/train_7b_attnres_4node.sh start
bash scripts/train_7b_dar_4node.sh start

bash scripts/train_7b_dar_4node.sh status
bash scripts/train_7b_dar_4node.sh logs
bash scripts/train_7b_dar_4node.sh stop
```

State the 40,000-step/20.97B-token arithmetic and require a one-iteration smoke
on each group with a separate `OUTPUT_DIR` before the production start.

- [ ] **Step 10: Run the full regression suite and commit Task 2**

Run:

```bash
python -m unittest discover -s tests -v
git diff --check
```

Expected: zero failures and no whitespace errors. Then commit only intended
files:

```bash
git add scripts/launch_7b_ssh_common.sh \
  scripts/train_7b_baseline_4node.sh \
  scripts/train_7b_attnres_4node.sh \
  scripts/train_7b_dar_4node.sh \
  tests/test_three_cluster_7b_launchers.py README.md
git commit -m "feat: launch three 7b four-node jobs"
```

---

## Final verification checklist

- [ ] Run `python -m unittest tests.test_7b_multi_node_overrides tests.test_three_cluster_7b_launchers -v` on committed HEAD.
- [ ] Run `python -m unittest discover -s tests -v` on committed HEAD and record the exact test count.
- [ ] Run `bash -n` over all seven changed/created shell scripts.
- [ ] Run each model entry with `DRY_RUN=1` against a temporary four-host file and inspect model/config/output separation.
- [ ] Confirm `git status --short` lists only the user's pre-existing unrelated changes.
- [ ] On the server, confirm both finalized token files and all three hostfiles, then run a one-iteration smoke on each cluster before starting production.
