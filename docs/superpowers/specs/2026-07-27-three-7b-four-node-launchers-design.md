# Three 7B four-node launchers design

## Goal

Launch three independent 7B MoE experiments—Baseline, Attention Residual, and
Dual Attention Residual—on three disjoint four-node/eight-GPU-per-node groups.
Each experiment trains for 40,000 iterations with global batch size 128 and a
2,000-iteration learning-rate warmup.

The shared server project root is:

```text
/gemini/space/internship/yuxingda/dual_attentions_residuals_megatron
```

## Files and interfaces

A common SSH fan-out launcher owns validation, remote environment construction,
`torch.distributed.run`, logging, and process supervision. Three small model
entry scripts provide only model-specific defaults:

- `scripts/launch_7b_ssh_common.sh`
- `scripts/train_7b_baseline_4node.sh` using `hostfile_baseline`
- `scripts/train_7b_attnres_4node.sh` using `hostfile_attnres`
- `scripts/train_7b_dar_4node.sh` using `hostfile_dar`

Each hostfile is stored at the repository root. It contains exactly four
non-empty, non-comment hostnames or IP addresses, one per line. The first entry
is rank 0 and the rendezvous master. Repeated hosts are rejected.

The three entry scripts start their controller in the background, write a PID
file, and return after startup. Each model has a separate controller log, four
node logs, output directory, and PID file. Starting an already-running model is
rejected. A stale PID file is removed safely. The scripts expose `start`,
`status`, `logs`, and `stop` operations; `start` is the default.

## Training contract

All three experiments use their existing 7B YAML architecture:

- 4 nodes × 8 GPUs = 32 processes
- TP / PP / EP / CP = 2 / 1 / 4 / 1
- sequence parallelism enabled
- DP world size = 16 (EP is nested within the data-parallel dimension)
- micro batch size 4, global batch size 128
- gradient accumulation = 2
- sequence length 4,096
- 40,000 training iterations
- BF16, distributed optimizer, selective recomputation
- peak/minimum learning rate `3e-4 / 3e-5`
- 2,000 warmup iterations followed by cosine decay through iteration 40,000
- checkpoint and validation every 10,000 iterations
- 20 validation iterations
- learned MoE routing (`MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0`)

This schedule processes 524,288 token positions per iteration and
20,971,520,000 token positions in total.

The launchers use only finalized inputs:

```text
data/fineweb_100BT_pretok/train.bin
data/fineweb_100BT_pretok/val.bin
```

They never fall back to `.partial` files. Existing environment overrides may
change paths and operational values, but every resolved value is validated and
printed before launch.

## Remote execution

The controller uses passwordless SSH rather than MPI. It runs a preflight on
all four hosts before launching any rank. Preflight verifies:

- the shared project directory;
- executable `.venv-train/bin/python`;
- the selected YAML and finalized token files;
- eight visible GPUs;
- importability of PyTorch, Megatron-Core, Transformer Engine, and Triton.

After all preflights pass, one SSH session per node executes the existing
model-specific multi-node script. Remote variables set `NNODES=4`, a unique
`NODE_RANK`, the first host as `MASTER_ADDR`, eight GPUs per node, the common
schedule, data paths, and output directory.

The remote environment rebuilds `LD_LIBRARY_PATH` from CUDA 13.2, the virtual
environment's NVIDIA wheels, and PyTorch libraries to avoid mixed CUDA 12/13
runtimes. NCCL enables IB and logs initialization/network diagnostics.

## Failure behavior

No training rank starts unless all four nodes pass preflight. If a remote rank
exits nonzero, the controller reports the failed host and returns nonzero. Stop
sends `TERM` to the controller, which in turn terminates its active SSH
children; `torchrun` propagates termination to local workers.

Output directories are model-specific and a non-empty checkpoint directory is
not silently overwritten. An explicit resume path is outside this launcher
task because distributed-optimizer resume is under separate repair.

## Verification

Automated tests use fake SSH and temporary hostfiles to verify:

- exactly four unique hosts are required;
- each rank receives the correct node rank, master address, model config, data
  paths, 40,000 iterations, batch sizes, and scheduler settings;
- Baseline, AttnRes, and DAR select separate configs, hostfiles, outputs, logs,
  and PID files;
- preflight failure launches no training nodes;
- background lifecycle commands handle running and stale PIDs safely;
- shell syntax and dry-run output are valid.

Target-cluster verification remains a one-iteration dry run followed by a short
real smoke for each group before starting all 40,000-iteration jobs.
