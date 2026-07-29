from pathlib import Path


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
