from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.tokenizer import SentencePieceTokenizer, TokenizerLike

FORMAT_VERSION = 1
TOKEN_DTYPE = "<i4"
EXPECTED_VOCAB_SIZE = 131_072

_WORKER_TOKENIZER: TokenizerLike | None = None


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


@dataclass(frozen=True)
class WorkerTask:
    source: FileIdentity
    shard_path: str
    sidecar_path: str
    tokenizer_sha256: str
    text_column: str
    batch_size: int
    add_bos: bool
    add_eos: bool


@dataclass(frozen=True)
class ShardResult:
    source: FileIdentity
    shard_path: str
    sidecar_path: str
    documents: int
    skipped_documents: int
    tokens: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "ShardResult":
        return cls(
            source=FileIdentity(**dict(payload["source"])),
            shard_path=str(payload["shard_path"]),
            sidecar_path=str(payload["sidecar_path"]),
            documents=int(payload["documents"]),
            skipped_documents=int(payload["skipped_documents"]),
            tokens=int(payload["tokens"]),
        )


def initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    pa.set_cpu_count(1)
    _WORKER_TOKENIZER = SentencePieceTokenizer(tokenizer_path)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_valid_shard(task: WorkerTask) -> ShardResult | None:
    shard = Path(task.shard_path)
    sidecar = Path(task.sidecar_path)
    if not shard.is_file() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"task", "result"}:
            return None
        if (
            not isinstance(payload["task"], dict)
            or canonical_json(payload["task"]) != canonical_json(asdict(task))
            or not isinstance(payload["result"], dict)
        ):
            return None
        raw_result = payload["result"]
        if (
            set(raw_result) != {
                "source", "shard_path", "sidecar_path", "documents", "skipped_documents", "tokens"
            }
            or canonical_json(raw_result["source"]) != canonical_json(asdict(task.source))
            or canonical_json(raw_result["shard_path"]) != canonical_json(task.shard_path)
            or canonical_json(raw_result["sidecar_path"]) != canonical_json(task.sidecar_path)
            or any(
                type(raw_result[field]) is not int
                for field in ("documents", "skipped_documents", "tokens")
            )
        ):
            return None
        result = ShardResult.from_json(raw_result)
        if (
            canonical_json(raw_result) != canonical_json(result.to_json())
            or result.source != task.source
            or result.shard_path != task.shard_path
            or result.sidecar_path != task.sidecar_path
            or result.documents < 0
            or result.skipped_documents < 0
            or result.tokens < 0
            or shard.stat().st_size != result.tokens * np.dtype(TOKEN_DTYPE).itemsize
        ):
            return None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return result


def tokenize_file(task: WorkerTask) -> ShardResult:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    shard = Path(task.shard_path)
    sidecar = Path(task.sidecar_path)
    shard.parent.mkdir(parents=True, exist_ok=True)
    partial = shard.with_suffix(shard.suffix + ".partial")
    partial.unlink(missing_ok=True)
    documents = skipped = tokens = 0
    try:
        parquet = pq.ParquetFile(task.source.path)
        if task.text_column not in parquet.schema_arrow.names:
            raise ValueError(f"missing text column {task.text_column!r}: {task.source.path}")
        with partial.open("wb") as output:
            for batch in parquet.iter_batches(
                batch_size=task.batch_size, columns=[task.text_column], use_threads=False
            ):
                values = batch.column(0).to_pylist()
                texts = [value for value in values if isinstance(value, str) and value.strip()]
                skipped += len(values) - len(texts)
                encoded = _WORKER_TOKENIZER.encode_batch(
                    texts, add_bos=task.add_bos, add_eos=task.add_eos
                )
                for token_ids in encoded:
                    if not token_ids:
                        skipped += 1
                        continue
                    if min(token_ids) < -(2**31) or max(token_ids) >= 2**31:
                        raise OverflowError(f"token ID outside int32 range in {task.source.path}")
                    array = np.asarray(token_ids, dtype=TOKEN_DTYPE)
                    array.tofile(output)
                    tokens += int(array.size)
                    documents += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, shard)
        result = ShardResult(task.source, str(shard), str(sidecar), documents, skipped, tokens)
        atomic_write_json(sidecar, {"task": asdict(task), "result": result.to_json()})
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def make_worker_tasks(run: RunIdentity, state_dir: Path, batch_size: int) -> list[WorkerTask]:
    shard_dir = state_dir / "shards"
    return [
        WorkerTask(
            source=source,
            shard_path=str(shard_dir / f"{source.index:06d}.bin"),
            sidecar_path=str(shard_dir / f"{source.index:06d}.json"),
            tokenizer_sha256=run.tokenizer_sha256,
            text_column=run.text_column,
            batch_size=batch_size,
            add_bos=run.add_bos,
            add_eos=run.add_eos,
        )
        for source in run.inputs
    ]


def summarize_progress(
    done: Sequence[ShardResult], total: int, started_at: float, now: float | None = None
) -> str:
    current = time.monotonic() if now is None else now
    elapsed = max(current - started_at, 1e-9)
    documents = sum(item.documents for item in done)
    tokens = sum(item.tokens for item in done)
    return (
        f"files={len(done)}/{total} documents={documents:,} tokens={tokens:,} "
        f"elapsed={elapsed:.1f}s tokens/s={tokens / elapsed:,.1f}"
    )


def _validate_scheduler_inputs(tasks: Sequence[WorkerTask], num_workers: int) -> None:
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers must be positive")
    indexes = [task.source.index for task in tasks]
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate source indexes are not allowed")


def _validate_heartbeat(heartbeat_seconds: float) -> None:
    if (
        isinstance(heartbeat_seconds, bool)
        or not isinstance(heartbeat_seconds, (int, float))
        or not math.isfinite(heartbeat_seconds)
        or heartbeat_seconds <= 0
    ):
        raise ValueError("heartbeat_seconds must be a finite positive number")


def _run_pending_tasks(
    tasks: Sequence[WorkerTask],
    tokenizer_path: str,
    num_workers: int,
    already_done: Sequence[ShardResult],
    total: int,
    started_at: float,
    heartbeat_seconds: float,
    emit: Callable[[str], None],
) -> list[ShardResult]:
    _validate_scheduler_inputs(tasks, num_workers)
    _validate_heartbeat(heartbeat_seconds)
    completed_results = list(already_done)
    results: list[ShardResult] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=get_context("spawn"),
        initializer=initialize_worker,
        initargs=(tokenizer_path,),
    ) as executor:
        futures = {executor.submit(tokenize_file, task): task for task in tasks}
        pending = set(futures)
        while pending:
            completed, pending = wait(
                pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED
            )
            if not completed:
                emit(summarize_progress(completed_results, total, started_at))
                continue
            for future in completed:
                task = futures[future]
                try:
                    result = future.result()
                except BaseException as error:
                    for remaining in pending:
                        remaining.cancel()
                    raise RuntimeError(f"failed to preprocess {task.source.path}") from error
                results.append(result)
                completed_results.append(result)
            emit(summarize_progress(completed_results, total, started_at))
    return results


def process_shards(
    tasks: Sequence[WorkerTask],
    tokenizer_path: str,
    num_workers: int,
    resume: bool,
    heartbeat_seconds: float = 30.0,
    emit: Callable[[str], None] = print,
) -> list[ShardResult]:
    _validate_scheduler_inputs(tasks, num_workers)
    _validate_heartbeat(heartbeat_seconds)
    started_at = time.monotonic()
    reused: list[ShardResult] = []
    pending: list[WorkerTask] = []
    for task in tasks:
        existing = load_valid_shard(task) if resume else None
        if existing is None:
            pending.append(task)
        else:
            reused.append(existing)
    emit(f"files={len(tasks)} reused={len(reused)} pending={len(pending)}")
    built = (
        _run_pending_tasks(
            pending,
            tokenizer_path,
            num_workers,
            reused,
            len(tasks),
            started_at,
            heartbeat_seconds,
            emit,
        )
        if pending
        else []
    )
    results = sorted([*reused, *built], key=lambda item: item.source.index)
    expected_indexes = [task.source.index for task in tasks]
    actual_indexes = [item.source.index for item in results]
    if len(actual_indexes) != len(set(actual_indexes)) or actual_indexes != expected_indexes:
        raise RuntimeError(
            f"shard results do not match input tasks: expected {expected_indexes}, got {actual_indexes}"
        )
    return results


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
