from pathlib import Path
from unittest import mock

from data import download


def test_default_arguments() -> None:
    args = download.parse_args([])
    assert args.output_dir == Path("data/fineweb_edu_100BT")
    assert args.num_workers == 8


def test_main_downloads_only_fineweb_edu_100bt(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    with mock.patch.object(
        download, "snapshot_download", return_value=str(destination)
    ) as snapshot_download:
        assert download.main(["--output-dir", str(destination), "--num-workers", "3"]) == 0
    snapshot_download.assert_called_once_with(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns=["sample/100BT/*.parquet"],
        local_dir=destination,
        max_workers=3,
    )
