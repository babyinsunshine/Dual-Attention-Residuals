from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

FORMAT_VERSION = 1
TOKEN_DTYPE = "<i4"
EXPECTED_VOCAB_SIZE = 131_072


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel local-Parquet tokenizer for FineWeb-Edu.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-workers", type=positive_int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=positive_int, default=1_024)
    parser.add_argument("--validation-tokens", type=positive_int, default=10_000_000)
    parser.add_argument("--max-files", type=positive_int)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--add-eos", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class FileIdentity:
    index: int
    path: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, index: int, path: Path) -> "FileIdentity":
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file():
            raise ValueError(f"input is not a regular file: {resolved}")
        return cls(index=index, path=str(resolved), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


@dataclass(frozen=True)
class RunIdentity:
    format_version: int
    inputs: tuple[FileIdentity, ...]
    tokenizer_path: str
    tokenizer_sha256: str
    vocab_size: int
    text_column: str
    add_bos: bool
    add_eos: bool

    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def discover_inputs(pattern: str, max_files: int | None) -> list[Path]:
    paths = sorted(
        path.resolve()
        for raw in glob.glob(pattern)
        if (path := Path(raw)).is_file() and path.suffix == ".parquet"
    )
    if not paths:
        raise ValueError(f"input glob matched no Parquet files: {pattern}")
    return paths if max_files is None else paths[:max_files]


def build_run_identity(
    inputs: Sequence[Path], tokenizer_path: Path, vocab_size: int,
    text_column: str, add_bos: bool, add_eos: bool,
) -> RunIdentity:
    resolved_tokenizer = tokenizer_path.resolve(strict=True)
    return RunIdentity(
        format_version=FORMAT_VERSION,
        inputs=tuple(FileIdentity.from_path(i, path) for i, path in enumerate(inputs)),
        tokenizer_path=str(resolved_tokenizer),
        tokenizer_sha256=sha256(resolved_tokenizer),
        vocab_size=vocab_size,
        text_column=text_column,
        add_bos=add_bos,
        add_eos=add_eos,
    )
