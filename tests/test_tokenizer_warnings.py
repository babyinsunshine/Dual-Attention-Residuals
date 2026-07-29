from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_unrelated_deprecation_warning_remains_visible() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import warnings\n"
                "import tests.conftest\n"
                "import data.tokenizer\n"
                "warnings.warn('unrelated tokenizer deprecation', DeprecationWarning)\n"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "unrelated tokenizer deprecation" in completed.stderr
    assert "SwigPy" not in completed.stderr
    assert "swigvarlink" not in completed.stderr
