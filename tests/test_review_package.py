from pathlib import Path
import re
import subprocess
import sys

import tomli
import yaml

from data import preprocess
from dual_attentions_residuals.config import TrainConfig


ROOT = Path(__file__).resolve().parents[1]

MARKDOWN_LINK_PATTERNS = (
    re.compile(r"!?\[[^\]\r\n]*\]\([^\)\r\n]*\)"),
    re.compile(r"!?\[[^\]\r\n]*\]\[[^\]\r\n]*\]"),
    re.compile(r"^\s*\[[^\]\r\n]+\]:\s*\S+", flags=re.MULTILINE),
)


def contains_markdown_link_or_image(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in MARKDOWN_LINK_PATTERNS)


def test_training_surface_contains_only_three_review_launchers() -> None:
    scripts = {path.name for path in (ROOT / "scripts").glob("train_*.sh")}
    assert scripts == {
        "train_baseline_single_node.sh",
        "train_attnres_single_node.sh",
        "train_dual_single_node.sh",
    }


def test_configuration_surface_contains_only_three_7b_models() -> None:
    configs = {path.name for path in (ROOT / "configs").glob("*.yaml")}
    assert configs == {
        "7b_moe_baseline.yaml",
        "7b_moe_attnres.yaml",
        "7b_moe_dual.yaml",
    }


def test_scripts_contain_no_cluster_orchestration_artifacts() -> None:
    names = {path.name for path in (ROOT / "scripts").iterdir() if path.is_file()}
    assert not any(
        token in name
        for name in names
        for token in ("multi_node", "4node", "ssh", "hostfile")
    )


def test_readme_is_anonymous_and_link_free() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"https?://|www\.", text, flags=re.IGNORECASE) is None
    assert not contains_markdown_link_or_image(text)
    assert re.search(r"/(?:home|root|gfs|gemini|workspace)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None


def test_readme_link_policy_recognizes_inline_and_reference_forms() -> None:
    forbidden_examples = (
        "[guide](docs/guide.md)",
        "![plot](plot.png)",
        "[guide][documentation]\n[documentation]: docs/guide.md",
        "![plot][figure]\n[figure]: plot.png",
    )
    assert all(contains_markdown_link_or_image(example) for example in forbidden_examples)


def test_readme_documents_only_public_commands() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for command in (
        "python data/download.py",
        "python data/preprocess.py",
        "bash scripts/train_baseline_single_node.sh",
        "bash scripts/train_attnres_single_node.sh",
        "bash scripts/train_dual_single_node.sh",
    ):
        assert command.lower() in text
    for forbidden in ("hostfile", "ssh", "multi-node", "multi node", "node_rank", "master_addr"):
        assert forbidden not in text
    assert "eos token by default" in text
    assert "--no-add-eos" in text


def test_processed_data_paths_are_consistent_across_all_surfaces() -> None:
    expected_dir = "data/fineweb_edu_100BT_pretok"
    expected_train = f"{expected_dir}/train.bin"
    expected_val = f"{expected_dir}/val.bin"

    assert preprocess.parse_args([]).output_dir == expected_dir
    defaults = TrainConfig()
    assert defaults.train_tokens_file == expected_train
    assert defaults.val_tokens_file == expected_val
    for config_path in sorted((ROOT / "configs").glob("*.yaml")):
        train = yaml.safe_load(config_path.read_text(encoding="utf-8"))["train"]
        assert train["train_tokens_file"] == expected_train, config_path
        assert train["val_tokens_file"] == expected_val, config_path

    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert "data/fineweb_edu_100BT/" in ignored
    assert "data/fineweb_edu_100BT_pretok/" in ignored


def test_dependency_metadata_matches_documented_runtime() -> None:
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = set(pyproject["project"]["dependencies"])
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for dependency in ("megatron-core==0.18.2", "huggingface-hub==0.36.2"):
        assert dependency in project_dependencies
        assert dependency in requirements
    assert not any(item.lower().startswith("datasets") for item in project_dependencies)
    assert not any(item.lower().startswith("datasets") for item in requirements)


def test_importing_tokenizer_does_not_mutate_global_warning_policy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import warnings\n"
                "before = list(warnings.filters)\n"
                "import data.tokenizer\n"
                "raise SystemExit(0 if warnings.filters == before else 1)\n"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
