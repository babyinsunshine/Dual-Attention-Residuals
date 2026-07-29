from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "attnres_kernel_mode.sh"

FUSED_FLAGS = {
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


class AttnResKernelModeTests(unittest.TestCase):
    def run_helper(self, mode: str | None) -> subprocess.CompletedProcess[str]:
        names = " ".join(FUSED_FLAGS)
        command = f"""
set -euo pipefail
source "$1"
configure_attnres_kernel_mode
printf 'ATTRES_KERNEL_MODE=%s\\n' "$ATTRES_KERNEL_MODE"
for name in {names}; do
  printf '%s=%s\\n' "$name" "${{!name}}"
done
"""
        env = os.environ.copy()
        for name in FUSED_FLAGS:
            env.pop(name, None)
        if mode is None:
            env.pop("ATTRES_KERNEL_MODE", None)
        else:
            env["ATTRES_KERNEL_MODE"] = mode
        return subprocess.run(
            ["bash", "-c", command, "bash", str(HELPER)],
            cwd=REPOSITORY,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def parse_environment(stdout: str) -> dict[str, str]:
        return dict(line.split("=", 1) for line in stdout.splitlines())

    def test_fused_is_default_and_exports_current_production_values(self) -> None:
        completed = self.run_helper(None)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        actual = self.parse_environment(completed.stdout)
        self.assertEqual(actual.pop("ATTRES_KERNEL_MODE"), "fused")
        self.assertEqual(actual, FUSED_FLAGS)

    def test_reference_disables_every_attnres_optimized_path(self) -> None:
        completed = self.run_helper("reference")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        actual = self.parse_environment(completed.stdout)
        self.assertEqual(actual.pop("ATTRES_KERNEL_MODE"), "reference")
        self.assertEqual(actual, dict.fromkeys(FUSED_FLAGS, "0"))

    def test_invalid_mode_is_rejected(self) -> None:
        completed = self.run_helper("typo")

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "ATTRES_KERNEL_MODE must be fused or reference, got: typo",
            completed.stderr,
        )

if __name__ == "__main__":
    unittest.main()
