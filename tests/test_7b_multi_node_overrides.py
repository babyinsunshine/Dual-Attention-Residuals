from __future__ import annotations

import os
import shlex
import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
EXPECTED = {
    "train_baseline_multi_node.sh": "configs/7b_moe_baseline.yaml",
    "train_attnres_multi_node.sh": "configs/7b_moe_attnres.yaml",
    "train_dual_multi_node.sh": "configs/7b_moe_dual.yaml",
}


class MultiNodeOverrideTests(unittest.TestCase):
    def test_launchers_forward_all_explicit_overrides(self) -> None:
        for launcher, config in EXPECTED.items():
            with self.subTest(launcher=launcher):
                env = os.environ.copy()
                env.update(
                    {
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
                    }
                )

                completed = subprocess.run(
                    ["bash", str(SCRIPTS / launcher)],
                    cwd=REPOSITORY,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(
                    shlex.split(completed.stdout),
                    [
                        "/opt/train/python",
                        "-m",
                        "torch.distributed.run",
                        "--nnodes=4",
                        "--nproc-per-node=8",
                        "--node-rank=2",
                        "--master-addr=10.0.0.1",
                        "--master-port=29600",
                        "train.py",
                        "--config",
                        config,
                        "--train-tokens-file",
                        "/data/train.bin",
                        "--val-tokens-file",
                        "/data/val.bin",
                        "--output-dir",
                        "/output/model",
                        "--train-iters",
                        "40000",
                        "--micro-batch-size",
                        "4",
                        "--global-batch-size",
                        "128",
                        "--save-interval",
                        "10000",
                        "--eval-interval",
                        "10000",
                        "--eval-iters",
                        "20",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
