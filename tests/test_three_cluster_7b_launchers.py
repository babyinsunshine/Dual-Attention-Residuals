from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
EXPECTED = {
    "train_7b_baseline_4node.sh": (
        "hostfile_baseline",
        "configs/7b_moe_baseline.yaml",
        "baseline",
        "train_baseline_multi_node.sh",
    ),
    "train_7b_attnres_4node.sh": (
        "hostfile_attnres",
        "configs/7b_moe_attnres.yaml",
        "attnres",
        "train_attnres_multi_node.sh",
    ),
    "train_7b_dar_4node.sh": (
        "hostfile_dar",
        "configs/7b_moe_dual.yaml",
        "dar",
        "train_dual_multi_node.sh",
    ),
}
HOSTS = ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4")


class ThreeClusterLauncherTests(unittest.TestCase):
    def make_runtime(
        self, root: Path, entry_name: str = "train_7b_baseline_4node.sh"
    ) -> tuple[dict[str, str], Path, Path, Path, Path]:
        hostfile_name, _config, model, _launcher = EXPECTED[entry_name]
        hostfile = root / hostfile_name
        hostfile.write_text("\n".join(HOSTS) + "\n")

        data_dir = root / "data" / "fineweb_100BT_pretok"
        data_dir.mkdir(parents=True)
        train_tokens = data_dir / "train.bin"
        val_tokens = data_dir / "val.bin"
        train_tokens.touch()
        val_tokens.touch()

        python_bin = root / "python"
        python_bin.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ $* == *'import site; print'* ]]; then
  echo "$FAKE_SITE_PACKAGES"
elif [[ $* == *'from pathlib import Path; import torch'* ]]; then
  echo "$FAKE_TORCH_LIB"
elif [[ $* == *'import torch; import megatron.core'* ]]; then
  if [[ -n ${FAKE_PREFLIGHT_LOG:-} ]]; then
    printf 'preflight\\t%s\\t%s\\n' "${CUDA_HOME:-missing}" \
      "${LD_LIBRARY_PATH:-missing}" >> "$FAKE_PREFLIGHT_LOG"
  fi
elif [[ ${1:-} == -m && ${2:-} == torch.distributed.run ]]; then
  if [[ -n ${FAKE_PYTHON_LOG:-} ]]; then
    printf 'launch\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \
      "${MEGATRON_LR_WARMUP_ITERS:-missing}" \
      "${MEGATRON_LR_DECAY_ITERS:-missing}" \
      "${MEGATRON_LOAD_CHECKPOINT:-missing}" \
      "${MEGATRON_RESUME_FROM:-missing}" "${DRY_RUN:-missing}" "$*" \
      >> "$FAKE_PYTHON_LOG"
  fi
  if [[ -n ${FAKE_REMOTE_PIDS:-} ]]; then
    printf '%s\\n' "$$" >> "$FAKE_REMOTE_PIDS"
  fi
  if [[ ${FAKE_PYTHON_BLOCK:-0} == 1 ]]; then
    trap 'exit 0' TERM INT
    while :; do sleep 0.05; done
  fi
  sleep 0.8
fi
exit 0
"""
        )
        python_bin.chmod(0o755)

        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_nvidia_smi = fake_bin / "nvidia-smi"
        fake_nvidia_smi.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' 0 1 2 3 4 5 6 7\n"
        )
        fake_nvidia_smi.chmod(0o755)

        cuda_home = root / "cuda-13.2"
        (cuda_home / "lib64").mkdir(parents=True)
        fake_site_packages = root / "site-packages"
        (fake_site_packages / "nvidia" / "fake-wheel" / "lib").mkdir(
            parents=True
        )
        fake_torch_lib = root / "torch" / "lib"
        fake_torch_lib.mkdir(parents=True)

        fake_ssh_calls = root / "ssh-calls.log"
        fake_ssh_pids = root / "ssh-pids.log"
        fake_ssh = root / "fake-ssh"
        fake_ssh.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
remote_script=$(mktemp)
trap 'rm -f "$remote_script"' EXIT
cat > "$remote_script"
args=("$@")
host=
host_index=-1
for ((i=0; i<${#args[@]}; i++)); do
  if [[ ${args[$i]} == root@* ]]; then
    host=${args[$i]#root@}
    host_index=$i
  fi
done
(( host_index >= 0 )) || { echo "missing remote host" >&2; exit 2; }
remote_command=
for ((i=host_index + 1; i<${#args[@]}; i++)); do
  [[ -z $remote_command ]] || remote_command+=' '
  remote_command+=${args[$i]}
done
eval "set -- $remote_command"
parsed=("$@")
mode=
mode_index=-1
for ((i=0; i<${#parsed[@]}; i++)); do
  if [[ ${parsed[$i]} == preflight || ${parsed[$i]} == launch || ${parsed[$i]} == terminate ]]; then
    mode=${parsed[$i]}
    mode_index=$i
  fi
done
[[ -n $mode ]] || { echo "missing remote mode" >&2; exit 2; }
if [[ $mode == preflight ]]; then
  printf 'preflight\\t%s\\t%s\\t%s\\t%s\\n' "$host" \
    "${parsed[$((mode_index + 4))]}" "${parsed[$((mode_index + 5))]}" \
    "${parsed[$((mode_index + 6))]}" >> "$FAKE_SSH_CALLS"
  [[ ${FAKE_SSH_FAIL_PREFLIGHT_HOST:-} != "$host" ]] || exit 23
elif [[ $mode == terminate ]]; then
  printf 'terminate\\t%s\\t%s\\t%s\\n' "$host" \
    "${parsed[$((mode_index + 1))]}" "${parsed[$((mode_index + 2))]}" \
    >> "$FAKE_SSH_CALLS"
  [[ ${FAKE_SSH_FAIL_TERMINATE_HOST:-} != "$host" ]] || exit 24
fi
if [[ ${FAKE_SSH_EXEC_REMOTE:-0} == 1 ]]; then
  bash -c "$remote_command" < "$remote_script"
  exit $?
fi
if [[ $mode == preflight || $mode == terminate ]]; then
  exit 0
fi
model=${parsed[$((mode_index + 8))]}
rank=${parsed[$((mode_index + 9))]}
master=${parsed[$((mode_index + 11))]}
printf 'launch\\t%s\\t%s\\t%s\\t%s\\n' "$host" "$rank" "$master" "$model" >> "$FAKE_SSH_CALLS"
printf '%s\\n' "$$" >> "$FAKE_SSH_PIDS"
printf 'fake ssh launched model=%s rank=%s host=%s\\n' "$model" "$rank" "$host"
[[ ${FAKE_SSH_FAIL_LAUNCH_RANK:-} != "$rank" ]] || exit 23
if [[ ${FAKE_SSH_DELAYED_FAIL_LAUNCH_RANK:-} == "$rank" ]]; then
  sleep 0.6
  exit 23
fi
if [[ ${FAKE_SSH_BLOCK:-0} == 1 ]]; then
  trap 'exit 0' TERM INT
  while :; do sleep 0.05; done
fi
sleep 0.8
"""
        )
        fake_ssh.chmod(0o755)

        output_dir = root / "outputs" / f"7b_moe_{model}_4node_40k"
        log_dir = root / "logs" / f"7b_moe_{model}_4node_40k"
        pid_file = log_dir / "controller.pid"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CUDA_HOME": str(cuda_home),
                "HOSTFILE": str(hostfile),
                "PYTHON_BIN": str(python_bin),
                "TRAIN_TOKENS_FILE": str(train_tokens),
                "VAL_TOKENS_FILE": str(val_tokens),
                "OUTPUT_DIR": str(output_dir),
                "LOG_DIR": str(log_dir),
                "PID_FILE": str(pid_file),
                "SSH_BIN": str(fake_ssh),
                "SSH_CONNECT_TIMEOUT": "1",
                "STARTUP_TIMEOUT": "5",
                "FAKE_SSH_CALLS": str(fake_ssh_calls),
                "FAKE_SSH_PIDS": str(fake_ssh_pids),
                "FAKE_REMOTE_PIDS": str(root / "remote-pids.log"),
                "FAKE_SITE_PACKAGES": str(fake_site_packages),
                "FAKE_TORCH_LIB": str(fake_torch_lib),
            }
        )
        return env, hostfile, fake_ssh_calls, log_dir, pid_file

    def run_entry(
        self,
        entry_name: str,
        env: dict[str, str],
        action: str | None = None,
        timeout: float = 10,
    ) -> subprocess.CompletedProcess[str]:
        command = ["bash", str(SCRIPTS / entry_name)]
        if action is not None:
            command.append(action)
        return subprocess.run(
            command,
            cwd=REPOSITORY,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def poll_status(
        self,
        entry_name: str,
        env: dict[str, str],
        expected: str,
        timeout: float = 5,
    ) -> subprocess.CompletedProcess[str]:
        deadline = time.monotonic() + timeout
        while True:
            completed = self.run_entry(entry_name, env, "status")
            if expected in completed.stdout:
                return completed
            if time.monotonic() >= deadline:
                self.fail(
                    f"status did not report {expected}: "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
            time.sleep(0.05)

    def poll_recorded_pids(
        self, env: dict[str, str], expected: int = 4, timeout: float = 5
    ) -> list[int]:
        pid_log = Path(env["FAKE_SSH_PIDS"])
        deadline = time.monotonic() + timeout
        while True:
            if pid_log.exists():
                pids = [int(line) for line in pid_log.read_text().splitlines()]
                if len(pids) >= expected:
                    return pids
            if time.monotonic() >= deadline:
                self.fail(f"fake SSH did not record {expected} launch PIDs")
            time.sleep(0.05)

    def poll_pid_file(
        self, path: Path, expected: int = 4, timeout: float = 5
    ) -> list[int]:
        deadline = time.monotonic() + timeout
        while True:
            if path.exists():
                pids = [int(line) for line in path.read_text().splitlines()]
                if len(pids) >= expected:
                    return pids
            if time.monotonic() >= deadline:
                self.fail(f"{path} did not record {expected} PIDs")
            time.sleep(0.05)

    def process_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file() and stat.read_text().split()[2] == "Z":
            return False
        return True

    def stop_test_processes(self, pids: list[int]) -> None:
        for pid in pids:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(
            self.process_exists(pid) for pid in pids
        ):
            time.sleep(0.05)

    def test_each_entry_dry_run_reports_literal_model_contract(self) -> None:
        seen_paths: set[tuple[str, str, str]] = set()
        for entry_name, (hostfile_name, config, model, launcher) in EXPECTED.items():
            with self.subTest(entry=entry_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env, hostfile, fake_calls, log_dir, pid_file = self.make_runtime(
                    root, entry_name
                )
                env.pop("OUTPUT_DIR")
                env.pop("LOG_DIR")
                env.pop("PID_FILE")
                env["DRY_RUN"] = "1"

                completed = self.run_entry(entry_name, env)

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"MODEL_NAME={model}", completed.stdout)
                self.assertIn(f"HOSTFILE={hostfile}", completed.stdout)
                self.assertEqual(hostfile.name, hostfile_name)
                self.assertIn(f"CONFIG_FILE={REPOSITORY / config}", completed.stdout)
                self.assertIn(
                    f"MODEL_LAUNCHER={SCRIPTS / launcher}", completed.stdout
                )
                for rank, host in enumerate(HOSTS):
                    self.assertIn(f"rank {rank}: {host}", completed.stdout)
                self.assertIn("MASTER_ADDR=10.0.0.1", completed.stdout)
                for setting in (
                    "NNODES=4",
                    "GPUS_PER_NODE=8",
                    "TRAIN_ITERS=40000",
                    "MICRO_BATCH_SIZE=4",
                    "GLOBAL_BATCH_SIZE=128",
                    "WARMUP_ITERS=2000",
                    "SAVE_INTERVAL=10000",
                    "EVAL_INTERVAL=10000",
                    "EVAL_ITERS=20",
                    "MEGATRON_FORCE_UNIFORM_MOE_ROUTING=0",
                ):
                    self.assertIn(setting, completed.stdout)
                self.assertIn(
                    f"TRAIN_TOKENS_FILE={root / 'data/fineweb_100BT_pretok/train.bin'}",
                    completed.stdout,
                )
                self.assertIn(
                    f"VAL_TOKENS_FILE={root / 'data/fineweb_100BT_pretok/val.bin'}",
                    completed.stdout,
                )
                expected_output = REPOSITORY / "outputs" / f"7b_moe_{model}_4node_40k"
                expected_log = REPOSITORY / "logs" / f"7b_moe_{model}_4node_40k"
                expected_pid = expected_log / "controller.pid"
                self.assertIn(f"OUTPUT_DIR={expected_output}", completed.stdout)
                self.assertIn(f"LOG_DIR={expected_log}", completed.stdout)
                self.assertIn(f"PID_FILE={expected_pid}", completed.stdout)
                self.assertFalse(fake_calls.exists())
                seen_paths.add(
                    (
                        next(
                            line
                            for line in completed.stdout.splitlines()
                            if line.startswith("OUTPUT_DIR=")
                        ),
                        f"LOG_DIR={expected_log}",
                        f"PID_FILE={expected_pid}",
                    )
                )
        self.assertEqual(len(seen_paths), 3)

    def test_each_entry_uses_its_literal_default_hostfile(self) -> None:
        for entry_name, (hostfile_name, _config, _model, _launcher) in EXPECTED.items():
            with self.subTest(entry=entry_name), tempfile.TemporaryDirectory() as tmp:
                env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                    Path(tmp), entry_name
                )
                env.pop("HOSTFILE")
                env["DRY_RUN"] = "1"

                completed = self.run_entry(entry_name, env)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    str(REPOSITORY / hostfile_name),
                    completed.stderr,
                )

    def test_hostfile_requires_exactly_four_unique_single_token_hosts(self) -> None:
        invalid_cases = {
            "three hosts": (
                "10.0.0.1\n10.0.0.2\n10.0.0.3\n",
                "hostfile must contain exactly 4 unique hosts",
            ),
            "five hosts": (
                "\n".join((*HOSTS, "10.0.0.5")) + "\n",
                "hostfile must contain exactly 4 unique hosts",
            ),
            "duplicate": (
                "10.0.0.1\n10.0.0.2\n10.0.0.1\n10.0.0.4\n",
                "duplicate host: 10.0.0.1",
            ),
            "extra token": (
                "10.0.0.1 slots=8\n10.0.0.2\n10.0.0.3\n10.0.0.4\n",
                "hostfile line must contain one host",
            ),
        }
        for label, (contents, error) in invalid_cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                env, hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(
                    Path(tmp)
                )
                hostfile.write_text(contents)
                env["DRY_RUN"] = "1"

                completed = self.run_entry("train_7b_baseline_4node.sh", env)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(error, completed.stderr)
                self.assertFalse(fake_calls.exists())

    def test_comments_and_blank_lines_do_not_change_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            hostfile.write_text(
                "# baseline cluster\n\n10.0.0.1 # rank zero\n10.0.0.2\n"
                "  10.0.0.3   # worker\n10.0.0.4\n"
            )
            env["DRY_RUN"] = "1"

            completed = self.run_entry("train_7b_baseline_4node.sh", env)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for rank, host in enumerate(HOSTS):
                self.assertIn(f"rank {rank}: {host}", completed.stdout)

    def test_missing_final_tokens_and_nonempty_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(root)
            Path(env["TRAIN_TOKENS_FILE"]).unlink()
            env["DRY_RUN"] = "1"

            missing = self.run_entry("train_7b_baseline_4node.sh", env)

            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("training tokens not found", missing.stderr)
            self.assertFalse(fake_calls.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(root)
            output_dir = Path(env["OUTPUT_DIR"])
            output_dir.mkdir(parents=True)
            (output_dir / "checkpoint.pt").touch()
            env["DRY_RUN"] = "1"

            nonempty = self.run_entry("train_7b_baseline_4node.sh", env)

            self.assertNotEqual(nonempty.returncode, 0)
            self.assertIn("output directory must be empty", nonempty.stderr)
            self.assertFalse(fake_calls.exists())

    def test_nonfinal_token_filename_and_invalid_batch_divisibility_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(root)
            partial = root / "train.partial.bin"
            partial.touch()
            env["TRAIN_TOKENS_FILE"] = str(partial)
            env["DRY_RUN"] = "1"

            completed = self.run_entry("train_7b_baseline_4node.sh", env)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("training tokens must be named train.bin", completed.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            env["GLOBAL_BATCH_SIZE"] = "127"
            env["DRY_RUN"] = "1"

            completed = self.run_entry("train_7b_baseline_4node.sh", env)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("GLOBAL_BATCH_SIZE must be divisible", completed.stderr)

    def test_controller_preflights_then_launches_all_ranks_with_distinct_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, fake_calls, log_dir, _pid_file = self.make_runtime(
                Path(tmp), "train_7b_dar_4node.sh"
            )

            completed = self.run_entry("train_7b_dar_4node.sh", env, "__controller")

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = fake_calls.read_text().splitlines() if fake_calls.exists() else []
            preflights = [line.split("\t") for line in calls if line.startswith("preflight\t")]
            launches = [line.split("\t") for line in calls if line.startswith("launch\t")]
            self.assertEqual(len(preflights), 4)
            self.assertEqual(len(launches), 4)
            preflight_indexes = [
                index for index, line in enumerate(calls) if line.startswith("preflight\t")
            ]
            launch_indexes = [
                index for index, line in enumerate(calls) if line.startswith("launch\t")
            ]
            self.assertLess(max(preflight_indexes), min(launch_indexes))
            node_ranks = sorted(call[2] for call in launches)
            self.assertEqual(node_ranks, ["0", "1", "2", "3"])
            self.assertEqual([call[3] for call in launches], ["10.0.0.1"] * 4)
            for rank in range(4):
                node_log = log_dir / f"dar_node{rank}.log"
                self.assertTrue(node_log.is_file())
                self.assertIn(f"model=dar rank={rank}", node_log.read_text())

    def test_preflight_failure_prevents_every_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_FAIL_PREFLIGHT_HOST"] = "10.0.0.3"

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertNotEqual(completed.returncode, 0)
            calls = fake_calls.read_text().splitlines() if fake_calls.exists() else []
            self.assertEqual(sum(line.startswith("preflight\t") for line in calls), 3)
            self.assertEqual(sum(line.startswith("launch\t") for line in calls), 0)

    def test_background_start_status_and_stop_terminate_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp), "train_7b_attnres_4node.sh"
            )
            env["FAKE_SSH_BLOCK"] = "1"
            entry = "train_7b_attnres_4node.sh"

            ssh_pids: list[int] = []
            try:
                started = self.run_entry(entry, env, "start")
                self.assertEqual(started.returncode, 0, started.stderr)
                running = self.poll_status(entry, env, "RUNNING")
                self.assertEqual(running.returncode, 0, running.stderr)
                controller_pid = int(pid_file.read_text().strip())
                ssh_pids = self.poll_recorded_pids(env)

                stopped = self.run_entry(entry, env, "stop", timeout=35)

                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.poll_status(entry, env, "STOPPED")
                self.assertFalse(self.process_exists(controller_pid))
                self.assertEqual(
                    [pid for pid in ssh_pids if self.process_exists(pid)], []
                )
            finally:
                self.stop_test_processes(ssh_pids)

    def test_stale_pid_reports_stopped_and_does_not_block_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_BLOCK"] = "1"
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("99999999\n")
            entry = "train_7b_baseline_4node.sh"

            status = self.run_entry(entry, env, "status")
            self.assertEqual(status.returncode, 1)
            self.assertIn("STOPPED", status.stdout)

            started = self.run_entry(entry, env, "start")
            self.assertEqual(started.returncode, 0, started.stderr)
            self.poll_status(entry, env, "RUNNING")
            stopped = self.run_entry(entry, env, "stop", timeout=35)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)

    def test_reused_live_pid_is_not_owned_or_signaled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp)
            )
            unrelated = subprocess.Popen(["setsid", "sleep", "30"])
            try:
                pid_file.parent.mkdir(parents=True)
                pid_file.write_text(f"{unrelated.pid}\n")
                Path(f"{pid_file}.owner").write_text("stale-owner-token\n")

                status = self.run_entry("train_7b_baseline_4node.sh", env, "status")
                stopped = self.run_entry("train_7b_baseline_4node.sh", env, "stop")

                self.assertEqual(status.returncode, 1)
                self.assertIn("STOPPED", status.stdout)
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.assertIsNone(unrelated.poll())
            finally:
                if unrelated.poll() is None:
                    unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_quick_launch_failure_never_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_BLOCK"] = "1"
            env["FAKE_SSH_FAIL_LAUNCH_RANK"] = "2"
            ssh_pids: list[int] = []
            try:
                started = self.run_entry(
                    "train_7b_baseline_4node.sh", env, "start", timeout=10
                )
                if Path(env["FAKE_SSH_PIDS"]).exists():
                    ssh_pids = [
                        int(line)
                        for line in Path(env["FAKE_SSH_PIDS"]).read_text().splitlines()
                    ]

                controller_log = Path(env["LOG_DIR"]) / "baseline_controller.log"
                diagnostic = (
                    controller_log.read_text() if controller_log.exists() else "missing log"
                )
                self.assertNotEqual(started.returncode, 0, diagnostic)
                self.assertIn(
                    "node rank 2 (10.0.0.3) exited during startup", diagnostic
                )
                self.assertFalse(Path(f"{pid_file}.ready").exists())
                status = self.run_entry("train_7b_baseline_4node.sh", env, "status")
                self.assertEqual(status.returncode, 1)
                self.assertIn("STOPPED", status.stdout)
            finally:
                self.stop_test_processes(ssh_pids)

    def test_remote_arguments_survive_login_shell_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            injected = root / "remote-command-injected"
            master = f"10.0.0.1;touch {injected};#"
            env["MASTER_ADDR"] = master

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(injected.exists())
            launches = [
                line.split("\t")
                for line in fake_calls.read_text().splitlines()
                if line.startswith("launch\t")
            ]
            self.assertEqual([call[3] for call in launches], [master] * 4)

    def test_smoke_overrides_reach_remote_scheduler_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            python_log = root / "python-launches.log"
            env.update(
                {
                    "FAKE_SSH_EXEC_REMOTE": "1",
                    "FAKE_PYTHON_LOG": str(python_log),
                    "TRAIN_ITERS": "1",
                    "WARMUP_ITERS": "0",
                }
            )

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            launches = [line.split("\t") for line in python_log.read_text().splitlines()]
            self.assertEqual(len(launches), 4)
            self.assertEqual([line[1] for line in launches], ["0"] * 4)
            self.assertEqual([line[2] for line in launches], ["1"] * 4)

    def test_pid_override_parent_is_created_before_controller_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            pid_file = root / "nested" / "controller-state" / "controller.pid"
            env["PID_FILE"] = str(pid_file)
            env["FAKE_SSH_BLOCK"] = "1"
            ssh_pids: list[int] = []
            try:
                started = self.run_entry("train_7b_baseline_4node.sh", env, "start")
                if Path(env["FAKE_SSH_PIDS"]).exists():
                    ssh_pids = [
                        int(line)
                        for line in Path(env["FAKE_SSH_PIDS"]).read_text().splitlines()
                    ]

                self.assertEqual(started.returncode, 0, started.stderr)
                self.assertTrue(pid_file.is_file())
                stopped = self.run_entry(
                    "train_7b_baseline_4node.sh", env, "stop", timeout=35
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
            finally:
                self.stop_test_processes(ssh_pids)

    def test_failed_launch_reports_rank_and_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_DELAYED_FAIL_LAUNCH_RANK"] = "2"

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("node rank 2 (10.0.0.3): FAILED", completed.stdout)

    def test_stop_explicitly_terminates_and_verifies_remote_owned_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, fake_calls, _log_dir, pid_file = self.make_runtime(root)
            env["FAKE_SSH_EXEC_REMOTE"] = "1"
            env["FAKE_PYTHON_BLOCK"] = "1"
            remote_pid_file = Path(env["FAKE_REMOTE_PIDS"])
            remote_pids: list[int] = []
            try:
                started = self.run_entry("train_7b_baseline_4node.sh", env, "start")
                self.assertEqual(started.returncode, 0, started.stderr)
                remote_pids = self.poll_pid_file(remote_pid_file)

                stopped = self.run_entry(
                    "train_7b_baseline_4node.sh", env, "stop", timeout=35
                )

                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                calls = fake_calls.read_text().splitlines()
                terminations = [
                    line.split("\t")
                    for line in calls
                    if line.startswith("terminate\t")
                ]
                self.assertEqual(len(terminations), 4)
                self.assertEqual({line[1] for line in terminations}, set(HOSTS))
                self.assertEqual(len({line[2] for line in terminations}), 4)
                self.assertEqual(len({line[3] for line in terminations}), 4)
                self.assertTrue(
                    all(line[2].startswith("/tmp/dual-attentions-baseline-") for line in terminations)
                )
                self.assertTrue(
                    all(line[3].endswith(f".rank{rank}") for rank, line in enumerate(terminations))
                )
                self.assertEqual(
                    [pid for pid in remote_pids if self.process_exists(pid)], []
                )
                self.assertFalse(pid_file.exists())
                self.assertFalse(Path(f"{pid_file}.owner").exists())
            finally:
                self.stop_test_processes(remote_pids)

    def test_failed_remote_termination_retains_lifecycle_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_BLOCK"] = "1"
            env["FAKE_SSH_FAIL_TERMINATE_HOST"] = "10.0.0.3"

            started = self.run_entry("train_7b_baseline_4node.sh", env, "start")
            self.assertEqual(started.returncode, 0, started.stderr)

            stopped = self.run_entry("train_7b_baseline_4node.sh", env, "stop")

            self.assertNotEqual(stopped.returncode, 0)
            self.assertIn("remote cleanup did not complete", stopped.stderr)
            self.assertTrue(pid_file.exists())
            self.assertTrue(Path(f"{pid_file}.owner").exists())
            self.assertTrue(Path(f"{pid_file}.cleanup_failed").exists())

    def test_startup_failure_retains_state_when_remote_cleanup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, _log_dir, pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_FAIL_LAUNCH_RANK"] = "2"
            env["FAKE_SSH_FAIL_TERMINATE_HOST"] = "10.0.0.3"

            started = self.run_entry("train_7b_baseline_4node.sh", env, "start")

            self.assertNotEqual(started.returncode, 0)
            self.assertIn("remote cleanup did not complete", started.stderr)
            self.assertTrue(pid_file.exists())
            self.assertTrue(Path(f"{pid_file}.owner").exists())
            self.assertTrue(Path(f"{pid_file}.cleanup_failed").exists())

    def test_concurrent_starts_publish_exactly_one_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, fake_calls, _log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            command = ["bash", str(SCRIPTS / "train_7b_baseline_4node.sh"), "start"]
            first = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            second = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            first_output = first.communicate(timeout=10)
            second_output = second.communicate(timeout=10)
            time.sleep(1)

            return_codes = [first.returncode, second.returncode]
            self.assertEqual(return_codes.count(0), 1, (first_output, second_output))
            self.assertEqual(
                return_codes.count(2), 1, (first_output, second_output)
            )
            calls = fake_calls.read_text().splitlines()
            self.assertEqual(sum(line.startswith("launch\t") for line in calls), 4)

    def test_remote_launch_clears_resume_controls_and_forces_real_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            python_log = root / "python-controls.log"
            env.update(
                {
                    "FAKE_SSH_EXEC_REMOTE": "1",
                    "FAKE_PYTHON_LOG": str(python_log),
                    "MEGATRON_LOAD_CHECKPOINT": "/unsafe/load",
                    "MEGATRON_RESUME_FROM": "/unsafe/resume",
                    "DRY_RUN": "1",
                }
            )

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            launches = (
                [line.split("\t") for line in python_log.read_text().splitlines()]
                if python_log.exists()
                else []
            )
            self.assertEqual(len(launches), 4)
            self.assertEqual([line[3] for line in launches], ["missing"] * 4)
            self.assertEqual([line[4] for line in launches], ["missing"] * 4)
            self.assertEqual([line[5] for line in launches], ["0"] * 4)

    def test_preflight_imports_use_the_same_clean_cuda_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            preflight_log = root / "preflight-imports.log"
            env.update(
                {
                    "FAKE_SSH_EXEC_REMOTE": "1",
                    "FAKE_PREFLIGHT_LOG": str(preflight_log),
                }
            )

            completed = self.run_entry(
                "train_7b_baseline_4node.sh", env, "__controller"
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            imports = [line.split("\t") for line in preflight_log.read_text().splitlines()]
            self.assertEqual(len(imports), 4)
            expected_ld = ":".join(
                (
                    str(Path(env["CUDA_HOME"]) / "lib64"),
                    str(
                        Path(env["FAKE_SITE_PACKAGES"])
                        / "nvidia"
                        / "fake-wheel"
                        / "lib"
                    ),
                    env["FAKE_TORCH_LIB"],
                    "/usr/local/nvidia/lib",
                    "/usr/local/nvidia/lib64",
                )
            )
            self.assertEqual([line[1] for line in imports], [env["CUDA_HOME"]] * 4)
            self.assertEqual([line[2] for line in imports], [expected_ld] * 4)

    def test_higher_rank_failure_interrupts_blocked_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, _hostfile, _fake_calls, log_dir, _pid_file = self.make_runtime(
                Path(tmp)
            )
            env["FAKE_SSH_BLOCK"] = "1"
            env["FAKE_SSH_DELAYED_FAIL_LAUNCH_RANK"] = "3"
            ssh_pids: list[int] = []
            try:
                started = self.run_entry("train_7b_baseline_4node.sh", env, "start")
                self.assertEqual(started.returncode, 0, started.stderr)
                ssh_pids = self.poll_recorded_pids(env)

                stopped = self.poll_status(
                    "train_7b_baseline_4node.sh", env, "STOPPED", timeout=3
                )

                self.assertEqual(stopped.returncode, 1)
                controller_log = log_dir / "baseline_controller.log"
                self.assertIn(
                    "node rank 3 (10.0.0.4): FAILED", controller_log.read_text()
                )
            finally:
                self.run_entry("train_7b_baseline_4node.sh", env, "stop", timeout=35)
                self.stop_test_processes(ssh_pids)

    def test_fixed_topology_and_schedule_bounds_are_enforced(self) -> None:
        invalid = (
            ({"GPUS_PER_NODE": "7"}, "GPUS_PER_NODE must be exactly 8"),
            ({"MASTER_PORT": "0"}, "MASTER_PORT must be in range 1..65535"),
            ({"MASTER_PORT": "65536"}, "MASTER_PORT must be in range 1..65535"),
            (
                {"TRAIN_ITERS": "10", "WARMUP_ITERS": "10"},
                "WARMUP_ITERS must satisfy 0 <= WARMUP_ITERS < TRAIN_ITERS",
            ),
            (
                {"TRAIN_ITERS": "10", "WARMUP_ITERS": "11"},
                "WARMUP_ITERS must satisfy 0 <= WARMUP_ITERS < TRAIN_ITERS",
            ),
        )
        for overrides, error in invalid:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                    Path(tmp)
                )
                env.update(overrides)
                env["DRY_RUN"] = "1"

                completed = self.run_entry("train_7b_baseline_4node.sh", env)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(error, completed.stderr)

    def test_stop_accepts_natural_controller_exit_before_group_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _hostfile, _fake_calls, _log_dir, _pid_file = self.make_runtime(
                root
            )
            bash_env = root / "delay-controller-kill.sh"
            bash_env.write_text(
                """kill() {
  if [[ ${1:-} == -TERM && ${2:-} == -- && ${3:-} == -* ]]; then
    sleep 0.5
  fi
  builtin kill "$@"
}
"""
            )

            started = self.run_entry("train_7b_baseline_4node.sh", env, "start")
            self.assertEqual(started.returncode, 0, started.stderr)
            env["BASH_ENV"] = str(bash_env)

            stopped = self.run_entry("train_7b_baseline_4node.sh", env, "stop")

            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertIn("STOPPED", stopped.stdout)


if __name__ == "__main__":
    unittest.main()
