# SSH Multi-node Baseline Launcher Design

## Goal

Provide one repository script that launches the Megatron-Core baseline training job from a controller node across four SSH-accessible GPU nodes, while retaining environment-variable overrides for shorter tests and other node sets.

The default run matches the requested four-node schedule:

- hosts: `10.244.57.135`, `10.244.77.160`, `10.244.64.90`, and `10.244.82.236`
- eight GPUs per node and 32 processes in total
- 60,000 optimizer steps
- micro batch size 4 and global batch size 128
- TP/PP/EP/CP inherited from `configs/7b_moe_baseline.yaml` as 2/1/4/1
- 2,000 warmup steps followed by cosine decay through step 60,000
- validation and checkpointing every 10,000 steps, with 20 validation iterations

With 32 processes and TP=2, the Megatron data-parallel size is 16. The requested batch sizes therefore produce two gradient-accumulation microsteps per optimizer step.

## Interface

Add `scripts/launch_baseline_ssh.sh`. Running it with no arguments uses the defaults above. The following environment variables override defaults without editing the script:

- topology: `HOSTS`, `MASTER_ADDR`, `MASTER_PORT`, `GPUS_PER_NODE`
- runtime: `PYTHON_BIN`, `CUDA_HOME`, `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`
- data and output: `TRAIN_TOKENS_FILE`, `VAL_TOKENS_FILE`, `OUTPUT_DIR`, `LOG_DIR`
- schedule: `TRAIN_ITERS`, `MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `WARMUP_ITERS`, `SAVE_INTERVAL`, `EVAL_INTERVAL`, `EVAL_ITERS`
- behavior: `DRY_RUN`

`HOSTS` is a whitespace-separated list ordered by node rank. `MASTER_ADDR` defaults to the first host. Paths default to the existing FineWeb smoke snapshot and the project-local `.venv-train` environment.

The normal background invocation is:

```bash
nohup bash scripts/launch_baseline_ssh.sh \
  > baseline_4node_launcher.log 2>&1 < /dev/null &
```

The script itself does not daemonize. Keeping backgrounding outside the script preserves a visible launcher PID and straightforward shell semantics.

## Launch Flow

1. Resolve the project root relative to the script, independent of the caller's current directory.
2. Validate the host list, numeric settings, required local files, Python executable, and SSH connectivity.
3. Start one SSH session per host concurrently. The array index supplies `NODE_RANK`.
4. On each remote host, resolve that node's Python site-packages, PyTorch library directory, and NVIDIA wheel library directories. Construct a clean `LD_LIBRARY_PATH` headed by the configured CUDA toolkit.
5. Export NCCL, CUDA, allocator, and learning-rate scheduler settings on the remote node.
6. Invoke `train.py` through `python -m torch.distributed.run`, using explicit CLI overrides for batch sizes, schedule, data paths, validation, checkpointing, and output directory.
7. Redirect each SSH session to a separate rank log under `LOG_DIR`.
8. Wait for all sessions, report each node's result, and exit nonzero if any node fails.

The script will set `NCCL_IB_DISABLE=0`. NCCL remains responsible for selecting a usable RDMA interface and may fall back to sockets. With `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET`, the per-node logs expose whether `NET/IB` or `NET/Socket` was selected.

## Error Handling

The launcher uses strict Bash mode. Preflight failures stop before any training process starts. Once the distributed job begins, every node is launched before the script waits, preventing a sequential SSH deadlock. A failed SSH or training process is recorded, but the launcher still waits for the remaining node sessions and returns an overall failure status.

The controller's launcher must remain alive because its SSH sessions own the remote commands. `nohup` protects the controller process from terminal disconnects; stopping the launcher is treated as stopping the distributed run.

## Verification

Verification does not start a real 32-GPU job. It consists of:

1. `bash -n scripts/launch_baseline_ssh.sh` for shell syntax.
2. A local test harness with a fake `ssh` executable to verify four concurrent node-rank commands, expected schedule flags, log creation, and aggregated failure status.
3. `DRY_RUN=1` to inspect the resolved topology and training settings without opening SSH sessions.

After deployment on the server, a short run should confirm the rank-zero summary contains `dp=16`, `micro=4`, `global=128`, and `grad_accum=2`. NCCL initialization logs should be checked for `NET/IB` before committing to the 60,000-step run.

## Scope Boundary

This change only adds the SSH launcher and its launcher-level tests/documentation. It does not modify model architecture, the baseline YAML, preprocessing, checkpoint format, or dataset sampling behavior.
