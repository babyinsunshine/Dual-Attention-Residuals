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
        python_bin.write_text("#!/usr/bin/env bash\nexit 0\n")
        python_bin.chmod(0o755)

        fake_ssh_calls = root / "ssh-calls.log"
        fake_ssh = root / "fake-ssh"
        fake_ssh.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
args=("$@")
mode=
mode_index=-1
host=
for ((i=0; i<${#args[@]}; i++)); do
  [[ ${args[$i]} == root@* ]] && host=${args[$i]#root@}
  if [[ ${args[$i]} == preflight || ${args[$i]} == launch ]]; then
    mode=${args[$i]}
    mode_index=$i
  fi
done
[[ -n $mode ]] || { echo "missing remote mode" >&2; exit 2; }
if [[ $mode == preflight ]]; then
  printf 'preflight\\t%s\\n' "$host" >> "$FAKE_SSH_CALLS"
  [[ ${FAKE_SSH_FAIL_PREFLIGHT_HOST:-} != "$host" ]] || exit 23
  exit 0
fi
model=${args[$((mode_index + 8))]}
rank=${args[$((mode_index + 9))]}
master=${args[$((mode_index + 11))]}
printf 'launch\\t%s\\t%s\\t%s\\t%s\\n' "$host" "$rank" "$master" "$model" >> "$FAKE_SSH_CALLS"
printf 'fake ssh launched model=%s rank=%s host=%s\\n' "$model" "$rank" "$host"
if [[ ${FAKE_SSH_BLOCK:-0} == 1 ]]; then
  trap 'exit 0' TERM INT
  while :; do sleep 0.05; done
fi
sleep 0.3
"""
        )
        fake_ssh.chmod(0o755)

        output_dir = root / "outputs" / f"7b_moe_{model}_4node_40k"
        log_dir = root / "logs" / f"7b_moe_{model}_4node_40k"
        pid_file = log_dir / "controller.pid"
        env = os.environ.copy()
        env.update(
            {
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

            started = self.run_entry(entry, env, "start")
            self.assertEqual(started.returncode, 0, started.stderr)
            running = self.poll_status(entry, env, "RUNNING")
            self.assertEqual(running.returncode, 0, running.stderr)
            controller_pid = int(pid_file.read_text().strip())

            stopped = self.run_entry(entry, env, "stop", timeout=35)

            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.poll_status(entry, env, "STOPPED")
            with self.assertRaises(ProcessLookupError):
                os.kill(controller_pid, 0)

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


if __name__ == "__main__":
    unittest.main()
