from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "launch_baseline_ssh.sh"


class LauncherTests(unittest.TestCase):
    def make_runtime(self, root: Path) -> tuple[dict[str, str], Path, Path]:
        train_tokens = root / "train.bin"
        val_tokens = root / "val.bin"
        train_tokens.touch()
        val_tokens.touch()
        ssh_calls = root / "ssh-calls.log"
        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_SSH_CALLS"
cat >/dev/null
if [[ " $* " == *" launch "* ]] && [[ ${FAKE_SSH_FAIL_HOST:-} != "" ]] && [[ "$*" == *"root@$FAKE_SSH_FAIL_HOST"* ]]; then
  exit 23
fi
printf 'fake ssh completed: %s\\n' "$*"
"""
        )
        fake_ssh.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "PYTHON_BIN": "/bin/true",
                "TRAIN_TOKENS_FILE": str(train_tokens),
                "VAL_TOKENS_FILE": str(val_tokens),
                "OUTPUT_DIR": str(root / "output"),
                "LOG_DIR": str(root / "logs"),
                "FAKE_SSH_CALLS": str(ssh_calls),
                "SSH_CONNECT_TIMEOUT": "1",
            }
        )
        return env, ssh_calls, root / "logs"

    def run_launcher(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPOSITORY,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dry_run_reports_requested_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, ssh_calls, _logs = self.make_runtime(root)
            env["DRY_RUN"] = "1"
            completed = self.run_launcher(env)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for rank, host in enumerate(
                ("10.244.57.135", "10.244.77.160", "10.244.64.90", "10.244.82.236")
            ):
                self.assertIn(f"rank {rank}: {host}", completed.stdout)
            self.assertIn("TRAIN_ITERS=60000", completed.stdout)
            self.assertIn("MICRO_BATCH_SIZE=4", completed.stdout)
            self.assertIn("GLOBAL_BATCH_SIZE=128", completed.stdout)
            self.assertFalse(ssh_calls.exists())

    def test_fake_ssh_launches_all_four_ranks_and_creates_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, ssh_calls, logs = self.make_runtime(root)

            completed = self.run_launcher(env)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = ssh_calls.read_text().splitlines()
            self.assertEqual(sum(" preflight " in f" {line} " for line in calls), 4)
            self.assertEqual(sum(" launch " in f" {line} " for line in calls), 4)
            for rank in range(4):
                self.assertIn(f"node rank {rank}: OK", completed.stdout)
                node_log = logs / f"baseline_node{rank}.log"
                self.assertTrue(node_log.is_file())
                self.assertIn("fake ssh completed", node_log.read_text())
            self.assertIn("overall exit code: 0", completed.stdout)

    def test_fake_ssh_failure_is_aggregated_after_all_nodes_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, ssh_calls, _logs = self.make_runtime(root)
            env["FAKE_SSH_FAIL_HOST"] = "10.244.64.90"

            completed = self.run_launcher(env)

            self.assertEqual(completed.returncode, 1)
            calls = ssh_calls.read_text().splitlines()
            self.assertEqual(sum(" launch " in f" {line} " for line in calls), 4)
            self.assertIn("node rank 0: OK", completed.stdout)
            self.assertIn("node rank 1: OK", completed.stdout)
            self.assertIn("node rank 2: FAILED", completed.stdout)
            self.assertIn("node rank 3: OK", completed.stdout)
            self.assertIn("overall exit code: 1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
