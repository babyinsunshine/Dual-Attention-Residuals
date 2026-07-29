"""Download the FineWeb-Edu-100BT Parquet shards."""

import argparse
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_PATTERN = "sample/100BT/*.parquet"
DEFAULT_OUTPUT_DIR = Path("data/fineweb_edu_100BT")


def positive_int(value: str) -> int:
    """Return a positive integer accepted by argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse downloader command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu-100BT Parquet shards."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=positive_int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Download the selected FineWeb-Edu-100BT shards."""
    args = parse_args(argv)
    destination = snapshot_download(
        repo_id=DEFAULT_REPO_ID,
        repo_type="dataset",
        allow_patterns=[DEFAULT_PATTERN],
        local_dir=args.output_dir,
        max_workers=args.num_workers,
    )
    print(f"downloaded FineWeb-Edu-100BT to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
