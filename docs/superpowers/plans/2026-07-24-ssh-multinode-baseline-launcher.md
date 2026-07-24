# SSH Multi-node Baseline Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one tested Bash launcher that starts the requested 60,000-step Megatron-Core baseline run on four eight-GPU nodes through concurrent SSH sessions.

**Architecture:** The controller script resolves project-local defaults and environment overrides, performs read-only SSH preflight checks, then pipes a fixed remote Bash program to every host concurrently. Each SSH process owns one node's `torch.distributed.run` invocation and writes a separate log; the controller waits for all nodes and aggregates their exit status.

**Tech Stack:** Bash 4+, OpenSSH, Python 3.10 `unittest`, PyTorch distributed launcher, Megatron-Core/NCCL.

## Global Constraints

- Default hosts are `10.244.57.135 10.244.77.160 10.244.64.90 10.244.82.236`, in node-rank order.
- Default schedule is 60,000 steps, micro/global batch 4/128, 2,000 warmup steps, cosine decay through step 60,000, validation/checkpoint interval 10,000, and 20 validation iterations.
- Default topology is eight GPUs per node with TP/PP/EP/CP 2/1/4/1 inherited from `configs/7b_moe_baseline.yaml`.
- The script must not daemonize itself; users run it with `nohup`.
- Existing user changes in `dual_attentions_residuals/mcore_compat.py` and untracked benchmark configs must remain untouched.
- Tests must not contact real hosts or start GPU processes.

---

### Task 1: Define launcher behavior with subprocess tests

**Files:**
- Create: `tests/test_launch_baseline_ssh.py`
- Test: `tests/test_launch_baseline_ssh.py`

**Interfaces:**
- Consumes: executable path `scripts/launch_baseline_ssh.sh`; environment variables documented in the design.
- Produces: regression coverage for dry-run defaults, overrides, four-rank launch construction, and nonzero failure aggregation.

- [ ] **Step 1: Write a failing dry-run test**

Create a `unittest.TestCase` that invokes the missing launcher with `DRY_RUN=1`, a temporary log/output/data setup, and `/bin/true` as `PYTHON_BIN`. Assert output contains all four default hosts, `TRAIN_ITERS=60000`, `MICRO_BATCH_SIZE=4`, `GLOBAL_BATCH_SIZE=128`, and no real SSH call.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_launch_baseline_ssh.LauncherTests.test_dry_run_reports_requested_defaults -v
```

Expected: FAIL because `scripts/launch_baseline_ssh.sh` does not exist.

- [ ] **Step 3: Add fake-SSH success and failure tests**

The test fixture creates a temporary executable named `ssh` at the front of `PATH`. It records arguments and consumes the remote script from stdin. In success mode it exits zero; when `FAKE_SSH_FAIL_HOST` matches a host and the argument list contains `launch`, it exits 23. Tests assert:

```python
self.assertEqual(call_log.count(" mode=preflight"), 4)
self.assertEqual(call_log.count(" mode=launch"), 4)
self.assertIn("node rank 0: OK", completed.stdout)
self.assertEqual(failed.returncode, 1)
self.assertIn("node rank 2: FAILED", failed.stdout)
```

- [ ] **Step 4: Run the test module and confirm failures are caused by the missing implementation**

Run:

```bash
python -m unittest tests.test_launch_baseline_ssh -v
```

Expected: FAIL for the launcher tests because the production script is absent.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/test_launch_baseline_ssh.py
git commit -m "test: specify ssh multinode launcher"
```

### Task 2: Implement the SSH launcher

**Files:**
- Create: `scripts/launch_baseline_ssh.sh`
- Test: `tests/test_launch_baseline_ssh.py`

**Interfaces:**
- Consumes: whitespace-separated `HOSTS`; topology, runtime, data, output, and schedule environment variables from the design.
- Produces: one controller process, one `baseline_node<RANK>.log` per host, and exit status 0 only when every node succeeds.

- [ ] **Step 1: Add strict defaults and validation**

Implement strict Bash mode, resolve `PROJECT` relative to the script, parse `HOSTS` using `read -r -a HOST_ARRAY`, default `MASTER_ADDR` to the first host, and reject empty hosts or non-positive integer topology/schedule values. Resolve relative data/output/log paths beneath `PROJECT` without requiring those output directories to exist yet.

- [ ] **Step 2: Implement safe dry-run output**

When `DRY_RUN=1`, print the resolved project, ordered `rank -> host` mapping, topology, batch sizes, scheduler, data paths, output path, and log path; then exit before `ssh` is invoked.

- [ ] **Step 3: Implement preflight and concurrent launch**

For every host, pipe a quoted remote script into:

```bash
ssh -o BatchMode=yes -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
  "root@$host" bash -s -- preflight "$PROJECT" "$PYTHON_BIN" \
  "$TRAIN_TOKENS_FILE" "$VAL_TOKENS_FILE"
```

After all preflights pass, invoke the same remote script in `launch` mode concurrently. Its positional parameters carry node rank, node count, master address/port, GPU count, config/data/output paths, batch sizes, schedule, CUDA path, and NCCL settings. Avoid interpolating shell code from environment values.

The remote launch ends with:

```bash
exec "$python_bin" -m torch.distributed.run \
  --nnodes="$nnodes" \
  --nproc-per-node="$gpus_per_node" \
  --node-rank="$node_rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  train.py \
  --config "$config_file" \
  --train-tokens-file "$train_tokens_file" \
  --val-tokens-file "$val_tokens_file" \
  --output-dir "$output_dir" \
  --train-iters "$train_iters" \
  --micro-batch-size "$micro_batch_size" \
  --global-batch-size "$global_batch_size" \
  --save-interval "$save_interval" \
  --eval-interval "$eval_interval" \
  --eval-iters "$eval_iters"
```

- [ ] **Step 4: Aggregate node status**

Save each background SSH PID by rank, wait for all ranks even after a failure, print `node rank N: OK|FAILED`, print `overall exit code: 0|1`, and exit with that value.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
bash -n scripts/launch_baseline_ssh.sh
python -m unittest tests.test_launch_baseline_ssh -v
```

Expected: shell syntax succeeds and all launcher tests pass.

- [ ] **Step 6: Commit implementation**

```bash
git add scripts/launch_baseline_ssh.sh tests/test_launch_baseline_ssh.py
git commit -m "feat: add ssh multinode baseline launcher"
```

### Task 3: Document operation and perform final verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_launch_baseline_ssh.py`

**Interfaces:**
- Consumes: `scripts/launch_baseline_ssh.sh` behavior from Task 2.
- Produces: copyable dry-run, foreground diagnostic, background launch, monitoring, and IB verification commands.

- [ ] **Step 1: Add README usage**

Document:

```bash
DRY_RUN=1 bash scripts/launch_baseline_ssh.sh

nohup bash scripts/launch_baseline_ssh.sh \
  > baseline_4node_launcher.log 2>&1 < /dev/null &
echo $! > baseline_4node_launcher.pid

tail -f baseline_4node_launcher.log
tail -f logs/baseline_4node_60k/baseline_node0.log
grep -Eh 'NET/IB|NET/Socket|Using network|mlx5' \
  logs/baseline_4node_60k/baseline_node*.log | head -100
```

Include one override example with `TRAIN_ITERS=10 OUTPUT_DIR=outputs/baseline_4node_smoke`.

- [ ] **Step 2: Run complete verification**

Run:

```bash
bash -n scripts/launch_baseline_ssh.sh
python -m unittest tests.test_launch_baseline_ssh -v
git diff --check HEAD^ -- scripts/launch_baseline_ssh.sh tests/test_launch_baseline_ssh.py README.md
```

Expected: syntax exit 0, all tests pass, and no whitespace errors.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/superpowers/plans/2026-07-24-ssh-multinode-baseline-launcher.md
git commit -m "docs: explain ssh multinode training"
```
