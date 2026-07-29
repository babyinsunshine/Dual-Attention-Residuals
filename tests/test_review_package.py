from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


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
    assert re.search(r"/(?:home|root|gfs|gemini|workspace)/", text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text) is None


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
