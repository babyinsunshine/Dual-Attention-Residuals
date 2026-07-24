from __future__ import annotations

import argparse
import ctypes
import errno
import glob
import hashlib
import json
import math
import os
import shutil
import signal
import secrets
import stat
import sys
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

if not __package__:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from data.tokenizer import SentencePieceTokenizer, TokenizerLike

FORMAT_VERSION = 1
TOKEN_DTYPE = "<i4"
EXPECTED_VOCAB_SIZE = 131_072
TOKEN_BYTES = np.dtype(TOKEN_DTYPE).itemsize
COPY_TOKENS = 16_777_216
OWNED_FINAL_NAMES = (
    "train.bin", "val.bin", "metadata.json", "train.bin.partial", "val.bin.partial",
)

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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"expected a directory: {path}")
        return
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise RuntimeError(f"expected a directory: {current}")
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_directory(path.parent)
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
        _fsync_directory(path.parent)
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


@dataclass(frozen=True)
class RunManifest:
    sha256: str
    results: tuple[ShardResult, ...]


@dataclass(frozen=True)
class MergeState:
    manifest_sha256: str
    next_index: int
    validation_bytes: int
    training_bytes: int
    documents: int
    skipped_documents: int
    tokens: int


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
        _require_source_identity(task.source)
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
        _require_source_identity(task.source)
        os.replace(partial, shard)
        result = ShardResult(task.source, str(shard), str(sidecar), documents, skipped, tokens)
        atomic_write_json(sidecar, {"task": asdict(task), "result": result.to_json()})
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _require_source_identity(source: FileIdentity) -> None:
    """Require that a worker reads exactly the immutable input it was assigned."""
    try:
        resolved = Path(source.path).resolve(strict=True)
        status = resolved.stat()
    except OSError as error:
        raise RuntimeError(f"source identity changed: {source.path}") from error
    if (
        str(resolved) != source.path
        or not stat.S_ISREG(status.st_mode)
        or status.st_size != source.size
        or status.st_mtime_ns != source.mtime_ns
    ):
        raise RuntimeError(f"source identity changed: {source.path}")


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
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=get_context("spawn"),
        initializer=initialize_worker,
        initargs=(tokenizer_path,),
    )
    futures: dict[object, WorkerTask] = {}
    try:
        for task in tasks:
            futures[executor.submit(tokenize_file, task)] = task
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
                except Exception as error:
                    raise RuntimeError(f"failed to preprocess {task.source.path}") from error
                results.append(result)
                completed_results.append(result)
            emit(summarize_progress(completed_results, total, started_at))
    except BaseException:
        _abort_process_pool(executor, futures)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _abort_process_pool(executor: ProcessPoolExecutor, futures: Mapping[object, WorkerTask]) -> None:
    """Cancel queued work and terminate running spawned workers without waiting first.

    ``ProcessPoolExecutor`` on Python 3.10 has no public terminate-workers API;
    the private process map is used only here so SIGTERM/KeyboardInterrupt does
    not block in its context-manager ``shutdown(wait=True)`` path.
    """
    for future in futures:
        future.cancel()
    process_map = getattr(executor, "_processes", None)
    processes = tuple(process_map.values()) if isinstance(process_map, dict) else ()
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except (AttributeError, TypeError):  # pragma: no cover - Python < 3.9 fallback
        executor.shutdown(wait=False)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


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


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"invalid {field} in preprocessing state")
    return value


def _strict_file_identity(payload: object) -> FileIdentity:
    if not isinstance(payload, dict) or set(payload) != {"index", "path", "size", "mtime_ns"}:
        raise RuntimeError("invalid source identity in preprocessing state")
    index = _require_nonnegative_int(payload["index"], "source index")
    size = _require_nonnegative_int(payload["size"], "source size")
    mtime_ns = _require_nonnegative_int(payload["mtime_ns"], "source mtime_ns")
    if type(payload["path"]) is not str:
        raise RuntimeError("invalid source path in preprocessing state")
    return FileIdentity(index, payload["path"], size, mtime_ns)


def _strict_shard_result(payload: object) -> ShardResult:
    fields = {"source", "shard_path", "sidecar_path", "documents", "skipped_documents", "tokens"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise RuntimeError("invalid shard result in preprocessing state")
    if type(payload["shard_path"]) is not str or type(payload["sidecar_path"]) is not str:
        raise RuntimeError("invalid shard paths in preprocessing state")
    return ShardResult(
        _strict_file_identity(payload["source"]),
        payload["shard_path"],
        payload["sidecar_path"],
        _require_nonnegative_int(payload["documents"], "documents"),
        _require_nonnegative_int(payload["skipped_documents"], "skipped_documents"),
        _require_nonnegative_int(payload["tokens"], "tokens"),
    )


def _validate_results(results: Sequence[ShardResult]) -> tuple[ShardResult, ...]:
    normalized = tuple(results)
    if any(type(result) is not ShardResult for result in normalized):
        raise RuntimeError("run manifest contains an invalid shard result")
    indexes = [result.source.index for result in normalized]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise RuntimeError("run manifest shard indexes must be ordered and unique")
    for result in normalized:
        if (
            type(result.source) is not FileIdentity
            or type(result.source.index) is not int
            or type(result.source.path) is not str
            or type(result.source.size) is not int
            or type(result.source.mtime_ns) is not int
            or type(result.shard_path) is not str
            or type(result.sidecar_path) is not str
            or type(result.documents) is not int
            or type(result.skipped_documents) is not int
            or type(result.tokens) is not int
            or result.source.path == ""
            or result.source.index < 0
            or result.source.size < 0
            or result.source.mtime_ns < 0
            or result.documents < 0
            or result.skipped_documents < 0
            or result.tokens < 0
        ):
            raise RuntimeError("run manifest contains negative shard counters")
    return normalized


def _generated_shard_paths(state_dir: Path, source_index: int) -> tuple[Path, Path]:
    shard = state_dir / "shards" / f"{source_index:06d}.bin"
    return shard, shard.with_suffix(".json")


def _validate_generated_shard_paths(results: Sequence[ShardResult], state_dir: Path) -> None:
    resolved_state_dir = state_dir.resolve(strict=False)
    for result in results:
        expected_shard, expected_sidecar = _generated_shard_paths(resolved_state_dir, result.source.index)
        try:
            actual_shard = Path(result.shard_path).resolve(strict=False)
            actual_sidecar = Path(result.sidecar_path).resolve(strict=False)
        except OSError as error:
            raise RuntimeError("invalid generated shard paths") from error
        if actual_shard != expected_shard or actual_sidecar != expected_sidecar:
            raise RuntimeError("run manifest contains invalid generated shard paths")


def _manifest_payload(run_sha256: str, results: Sequence[ShardResult]) -> dict[str, object]:
    if type(run_sha256) is not str or not run_sha256:
        raise RuntimeError("run hash must be a non-empty string")
    normalized = _validate_results(results)
    return {"run_sha256": run_sha256, "results": [result.to_json() for result in normalized]}


def _atomic_write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_directory(path.parent)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
        ) as stream:
            partial = Path(stream.name)
            stream.write(canonical_json(payload).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except BaseException:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise


def _manifest_from_payload(payload: object, run_sha256: str) -> RunManifest:
    if not isinstance(payload, dict) or set(payload) != {"run_sha256", "results"}:
        raise RuntimeError("invalid run manifest structure")
    if type(payload["run_sha256"]) is not str or payload["run_sha256"] != run_sha256:
        raise RuntimeError("run hash does not match run manifest")
    if not isinstance(payload["results"], list):
        raise RuntimeError("invalid run manifest results")
    results = tuple(_strict_shard_result(item) for item in payload["results"])
    _validate_results(results)
    canonical = canonical_json({"run_sha256": run_sha256, "results": [item.to_json() for item in results]})
    return RunManifest(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), results)


def write_run_manifest(
    state_dir: Path, run_sha256: str, results: Sequence[ShardResult]
) -> RunManifest:
    payload = _manifest_payload(run_sha256, results)
    manifest = _manifest_from_payload(payload, run_sha256)
    _validate_generated_shard_paths(manifest.results, state_dir)
    path = state_dir / "run-manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid existing run manifest: {path}") from error
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError(f"existing run manifest differs: {path}")
    else:
        _atomic_write_canonical_json(path, payload)
    return manifest


def _validate_merge_state(payload: object, manifest: RunManifest) -> MergeState:
    fields = {
        "manifest_sha256", "next_index", "validation_bytes", "training_bytes",
        "documents", "skipped_documents", "tokens",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise RuntimeError("invalid merge state structure")
    if type(payload["manifest_sha256"]) is not str or payload["manifest_sha256"] != manifest.sha256:
        raise RuntimeError("merge state manifest hash does not match")
    state = MergeState(
        payload["manifest_sha256"],
        _require_nonnegative_int(payload["next_index"], "next_index"),
        _require_nonnegative_int(payload["validation_bytes"], "validation_bytes"),
        _require_nonnegative_int(payload["training_bytes"], "training_bytes"),
        _require_nonnegative_int(payload["documents"], "documents"),
        _require_nonnegative_int(payload["skipped_documents"], "skipped_documents"),
        _require_nonnegative_int(payload["tokens"], "tokens"),
    )
    if state.next_index > len(manifest.results):
        raise RuntimeError("merge state next_index is outside manifest bounds")
    if state.validation_bytes % TOKEN_BYTES or state.training_bytes % TOKEN_BYTES:
        raise RuntimeError("merge state byte counts are not int32 aligned")
    committed = manifest.results[:state.next_index]
    expected_documents = sum(item.documents for item in committed)
    expected_skipped = sum(item.skipped_documents for item in committed)
    expected_tokens = sum(item.tokens for item in committed)
    if (
        (state.documents, state.skipped_documents, state.tokens)
        != (expected_documents, expected_skipped, expected_tokens)
        or state.validation_bytes + state.training_bytes != expected_tokens * TOKEN_BYTES
    ):
        raise RuntimeError("merge state counters do not match committed shards")
    return state


def _load_merge_state(state_path: Path, manifest: RunManifest) -> MergeState:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid merge state: {state_path}") from error
    return _validate_merge_state(payload, manifest)


def load_resume_context(state_dir: Path, run_sha256: str) -> tuple[RunManifest, MergeState] | None:
    manifest_path = state_dir / "run-manifest.json"
    state_path = state_dir / "merge-state.json"
    manifest_exists, state_exists = manifest_path.exists(), state_path.exists()
    if not manifest_exists and not state_exists:
        return None
    if state_exists and not manifest_exists:
        raise RuntimeError("resume requires both run-manifest.json and merge-state.json")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid run manifest: {manifest_path}") from error
    manifest = _manifest_from_payload(manifest_payload, run_sha256)
    _validate_generated_shard_paths(manifest.results, state_dir)
    if not state_exists:
        # A crash after manifest publication but before the first journal entry
        # is a valid zero-commit recovery point.  Persist it before returning so
        # later recovery has one durable, validated interpretation.
        initial = _initial_merge_state(manifest)
        atomic_write_json(state_path, asdict(initial))
        return manifest, initial
    return manifest, _load_merge_state(state_path, manifest)


def recover_output_file(path: Path, committed_bytes: int) -> None:
    if type(committed_bytes) is not int or committed_bytes < 0 or committed_bytes % TOKEN_BYTES:
        raise RuntimeError(f"invalid committed byte count for merge output: {path}")
    _ensure_directory(path.parent)
    if not path.exists():
        if committed_bytes:
            raise RuntimeError(f"missing committed merge output: {path}")
        path.touch()
        _fsync_directory(path.parent)
    actual = path.stat().st_size
    if actual < committed_bytes:
        raise RuntimeError(
            f"merge output is shorter than journal: {path} ({actual} < {committed_bytes})"
        )
    if actual > committed_bytes:
        with path.open("r+b") as stream:
            stream.truncate(committed_bytes)
            stream.flush()
            os.fsync(stream.fileno())


def write_token_range(source: object, start: int, stop: int, destination: object) -> int:
    written = 0
    for offset in range(start, stop, COPY_TOKENS):
        end = min(stop, offset + COPY_TOKENS)
        chunk = np.asarray(source[offset:end], dtype=TOKEN_DTYPE)
        payload = chunk.tobytes(order="C")
        destination.write(payload)
        written += len(payload)
    return written


def _initial_merge_state(manifest: RunManifest) -> MergeState:
    return MergeState(manifest.sha256, 0, 0, 0, 0, 0, 0)


def merge_shards(
    results: Sequence[ShardResult], output_dir: Path, validation_tokens: int,
    run_manifest: RunManifest, resume: bool,
) -> MergeState:
    if type(validation_tokens) is not int or validation_tokens < 1:
        raise ValueError("validation_tokens must be positive")
    normalized = _validate_results(results)
    if type(run_manifest) is not RunManifest or normalized != run_manifest.results:
        raise RuntimeError("merge results do not match run manifest")
    state_dir = output_dir / ".preprocess"
    _validate_generated_shard_paths(normalized, state_dir)
    state_path = state_dir / "merge-state.json"
    validation_path = output_dir / "val.bin.partial"
    training_path = output_dir / "train.bin.partial"
    if resume:
        if not state_path.is_file():
            state = _initial_merge_state(run_manifest)
            atomic_write_json(state_path, asdict(state))
        else:
            state = _load_merge_state(state_path, run_manifest)
    else:
        if state_path.exists():
            raise RuntimeError(f"fresh merge refuses existing merge state: {state_path}")
        if validation_path.exists() or training_path.exists():
            raise RuntimeError("fresh merge refuses existing partial outputs")
        state = _initial_merge_state(run_manifest)
    committed_tokens = sum(item.tokens for item in normalized[:state.next_index])
    expected_validation_bytes = min(committed_tokens, validation_tokens) * TOKEN_BYTES
    if state.validation_bytes != expected_validation_bytes:
        raise RuntimeError("merge state validation bytes do not match the requested split")
    _ensure_directory(output_dir)
    recover_output_file(validation_path, state.validation_bytes)
    recover_output_file(training_path, state.training_bytes)
    for committed in normalized[:state.next_index]:
        shard, sidecar = _generated_shard_paths(state_dir, committed.source.index)
        shard.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    with validation_path.open("ab") as validation_output, training_path.open("ab") as training_output:
        for position, result in enumerate(normalized[state.next_index:], start=state.next_index):
            shard, sidecar = _generated_shard_paths(state_dir, result.source.index)
            expected_bytes = result.tokens * TOKEN_BYTES
            try:
                actual_bytes = shard.stat().st_size
            except OSError as error:
                raise RuntimeError(f"missing uncommitted merge shard: {shard}") from error
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"merge shard has unexpected byte size: {shard} ({actual_bytes} != {expected_bytes})"
                )
            source = (
                np.memmap(shard, dtype=TOKEN_DTYPE, mode="r", shape=(result.tokens,))
                if result.tokens
                else np.empty(0, dtype=TOKEN_DTYPE)
            )
            try:
                validation_remaining = max(0, validation_tokens - state.validation_bytes // TOKEN_BYTES)
                validation_count = min(validation_remaining, result.tokens)
                validation_written = write_token_range(source, 0, validation_count, validation_output)
                training_written = write_token_range(source, validation_count, result.tokens, training_output)
            finally:
                del source
            validation_output.flush()
            os.fsync(validation_output.fileno())
            training_output.flush()
            os.fsync(training_output.fileno())
            state = MergeState(
                run_manifest.sha256,
                position + 1,
                state.validation_bytes + validation_written,
                state.training_bytes + training_written,
                state.documents + result.documents,
                state.skipped_documents + result.skipped_documents,
                state.tokens + result.tokens,
            )
            atomic_write_json(state_path, asdict(state))
            shard.unlink()
            sidecar.unlink()
    return state


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


def _path_exists(path: Path) -> bool:
    """Return whether a directory entry exists, including a dangling symlink."""
    return os.path.lexists(path)


def _remove_tree_without_following_symlinks(path: Path) -> None:
    """Remove a generated state tree through no-follow directory descriptors."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            path.unlink()
            return
        raise
    try:
        _remove_directory_contents_without_following_symlinks(descriptor)
    finally:
        os.close(descriptor)
    path.rmdir()


def _remove_directory_contents_without_following_symlinks(directory_fd: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.scandir(os.dup(directory_fd)) as entries:
        for entry in entries:
            name = entry.name
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                if error.errno not in (errno.ENOTDIR, errno.ELOOP):
                    raise
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except IsADirectoryError as changed:
                    raise OSError(f"generated state entry changed while removing it: {name}") from changed
                continue
            try:
                _remove_directory_contents_without_following_symlinks(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)


def _remove_owned_outputs(output_dir: Path) -> None:
    changed = False
    for name in OWNED_FINAL_NAMES:
        candidate = output_dir / name
        if not _path_exists(candidate):
            continue
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            raise RuntimeError(f"owned output path is unexpectedly a directory: {candidate}")
        candidate.unlink()
        changed = True
    state_dir = output_dir / ".preprocess"
    if _path_exists(state_dir):
        if state_dir.parent != output_dir or state_dir.name != ".preprocess":
            raise RuntimeError(f"refusing to remove an unexpected preprocessing state directory: {state_dir}")
        _remove_tree_without_following_symlinks(state_dir)
        changed = True
    if changed:
        _fsync_directory(output_dir)


def prepare_output(output_dir: Path, overwrite: bool, resume: bool) -> Path:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    resolved = output_dir.resolve(strict=False)
    _ensure_directory(resolved)
    state_dir = resolved / ".preprocess"
    occupied = any(_path_exists(resolved / name) for name in OWNED_FINAL_NAMES)
    if occupied or _path_exists(state_dir):
        if overwrite:
            _remove_owned_outputs(resolved)
        elif not resume:
            raise FileExistsError(
                f"output already contains preprocessing files: {resolved}; use --resume or --overwrite"
            )
    return resolved


def _source_manifest_sha256(run: RunIdentity) -> str:
    payload = [asdict(source) for source in run.inputs]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


FileIdentityTuple = tuple[int, int, int]
RENAME_NOREPLACE = 1


def _rename_no_replace(
    source_directory_fd: int, source_name: str, destination_directory_fd: int, destination_name: str,
) -> None:
    """Linux renameat2(RENAME_NOREPLACE) anchored to already-open directories."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("atomic no-replace publication requires libc renameat2") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd, os.fsencode(source_name), destination_directory_fd,
        os.fsencode(destination_name), RENAME_NOREPLACE,
    )
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure == errno.EEXIST:
        raise FileExistsError(failure, os.strerror(failure), destination_name)
    raise OSError(failure, os.strerror(failure), destination_name)


def _fsync_directory_fd(directory_fd: int) -> None:
    os.fsync(directory_fd)


def _open_regular_file(
    directory_fd: int, name: str, expected_size: int | None, label: str, writable: bool = False,
) -> tuple[int, FileIdentityTuple]:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {name}")
        if expected_size is not None and status.st_size != expected_size:
            raise RuntimeError(
                f"{label} has unexpected byte size: {name} ({status.st_size} != {expected_size})"
            )
        return descriptor, (status.st_dev, status.st_ino, status.st_size)
    except BaseException:
        os.close(descriptor)
        raise


def _open_output_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _verify_named_identity(
    directory_fd: int, name: str, expected: FileIdentityTuple, label: str,
) -> None:
    descriptor, actual = _open_regular_file(directory_fd, name, expected[2], label)
    try:
        if actual != expected:
            raise RuntimeError(f"{label} changed after validation")
    finally:
        os.close(descriptor)


def _publish_binary_no_replace(
    directory_fd: int, partial_name: str, final_name: str, expected_bytes: int, label: str,
) -> tuple[int, FileIdentityTuple]:
    partial_fd, partial_identity = _open_regular_file(
        directory_fd, partial_name, expected_bytes, f"partial {label}"
    )
    final_fd: int | None = None
    published = False
    try:
        try:
            _rename_no_replace(directory_fd, partial_name, directory_fd, final_name)
            _fsync_directory_fd(directory_fd)
        except FileExistsError:
            final_fd, final_identity = _open_regular_file(directory_fd, final_name, expected_bytes, label)
            if final_identity != partial_identity:
                os.close(final_fd)
                final_fd = None
                raise RuntimeError(f"{label} destination already exists and has a different inode") from None
        else:
            final_fd, final_identity = _open_regular_file(directory_fd, final_name, expected_bytes, label)
            if final_identity != partial_identity:
                os.close(final_fd)
                final_fd = None
                raise RuntimeError(f"{label} publication did not preserve the validated partial inode")
        published = True
        return final_fd, final_identity
    finally:
        os.close(partial_fd)
        if final_fd is not None and not published:
            os.close(final_fd)


def _atomic_create_json(
    directory_fd: int, name: str, payload: Mapping[str, object],
) -> tuple[FileIdentityTuple, int]:
    """Atomically rename JSON into place and return its identity and writable fd."""
    temporary_name: str | None = None
    try:
        for _ in range(128):
            candidate = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.partial"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:  # pragma: no cover - cryptographic name collisions are infeasible
            raise RuntimeError("could not create unique completion metadata temporary")
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            status = os.fstat(stream.fileno())
            identity = (status.st_dev, status.st_ino, status.st_size)
        _rename_no_replace(directory_fd, temporary_name, directory_fd, name)
        temporary_name = None
        _fsync_directory_fd(directory_fd)
        metadata_fd, metadata_identity = _open_regular_file(
            directory_fd, name, identity[2], "completion metadata", writable=True
        )
        if metadata_identity != identity:
            os.close(metadata_fd)
            raise RuntimeError("completion metadata changed during publication")
        return identity, metadata_fd
    except BaseException:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise


def _invalidate_metadata_descriptor(descriptor: int) -> None:
    os.ftruncate(descriptor, 0)
    os.write(descriptor, b'{"incomplete":true}\n')
    os.fsync(descriptor)


def _final_metadata(
    run: RunIdentity, manifest: RunManifest, state: MergeState, batch_size: int, num_workers: int,
) -> dict[str, object]:
    validation_tokens = state.validation_bytes // TOKEN_BYTES
    training_tokens = state.training_bytes // TOKEN_BYTES
    return {
        "format_version": FORMAT_VERSION,
        "run_sha256": run.digest(),
        "manifest_sha256": manifest.sha256,
        "source_files": len(run.inputs),
        "source_manifest_sha256": _source_manifest_sha256(run),
        "documents": state.documents,
        "skipped_documents": state.skipped_documents,
        "train_tokens": training_tokens,
        "validation_tokens": validation_tokens,
        "total_tokens": training_tokens + validation_tokens,
        "dtype": "int32",
        "byte_order": "little",
        "tokenizer_path": run.tokenizer_path,
        "tokenizer_sha256": run.tokenizer_sha256,
        "vocab_size": run.vocab_size,
        "add_bos": run.add_bos,
        "add_eos": run.add_eos,
        "text_column": run.text_column,
        "num_workers": num_workers,
        "batch_size": batch_size,
    }


def _validate_finalization_inputs(
    run: RunIdentity, manifest: RunManifest, state: MergeState, batch_size: int, num_workers: int,
    validation_tokens: int,
) -> None:
    if type(run) is not RunIdentity or type(manifest) is not RunManifest or type(state) is not MergeState:
        raise RuntimeError("invalid preprocessing completion records")
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 1:
        raise ValueError("batch_size and num_workers must be positive")
    if type(validation_tokens) is not int or validation_tokens < 1:
        raise ValueError("validation_tokens must be positive")
    normalized_results = _validate_results(manifest.results)
    manifest_payload = {"run_sha256": run.digest(), "results": [item.to_json() for item in normalized_results]}
    expected_manifest_sha256 = hashlib.sha256(canonical_json(manifest_payload).encode("utf-8")).hexdigest()
    if type(manifest.sha256) is not str or manifest.sha256 != expected_manifest_sha256:
        raise RuntimeError("run manifest hash does not match the current run")
    _validate_merge_state(asdict(state), manifest)
    if state.manifest_sha256 != manifest.sha256 or state.next_index != len(manifest.results):
        raise RuntimeError("merge state does not match completed run manifest")
    if tuple(result.source for result in manifest.results) != run.inputs:
        raise RuntimeError("run manifest sources do not match the current run")
    expected_documents = sum(result.documents for result in manifest.results)
    expected_skipped_documents = sum(result.skipped_documents for result in manifest.results)
    expected_tokens = sum(result.tokens for result in manifest.results)
    if (state.documents, state.skipped_documents, state.tokens) != (
        expected_documents, expected_skipped_documents, expected_tokens,
    ):
        raise RuntimeError("merge state counters do not match completed run manifest")
    if state.validation_bytes % TOKEN_BYTES or state.training_bytes % TOKEN_BYTES:
        raise RuntimeError("merge state byte counts are not int32 aligned")
    if state.validation_bytes // TOKEN_BYTES != validation_tokens:
        raise RuntimeError("merge did not produce the requested validation token count")
    if state.training_bytes // TOKEN_BYTES <= 0:
        raise RuntimeError("merge did not produce positive training tokens")
    if state.tokens != (state.validation_bytes + state.training_bytes) // TOKEN_BYTES:
        raise RuntimeError("merge state token count does not match output byte counts")


def finalize_outputs(
    output_dir: Path, run: RunIdentity, manifest: RunManifest, merge_state: MergeState,
    batch_size: int, num_workers: int, validation_tokens: int,
) -> dict[str, object]:
    _validate_finalization_inputs(run, manifest, merge_state, batch_size, num_workers, validation_tokens)
    resolved = output_dir.resolve(strict=False)
    _ensure_directory(resolved)
    pairs = (
        ("val.bin.partial", "val.bin", merge_state.validation_bytes, "validation output"),
        ("train.bin.partial", "train.bin", merge_state.training_bytes, "training output"),
    )
    directory_fd = _open_output_directory(resolved)
    held_finals: list[tuple[int, FileIdentityTuple, str, str]] = []
    try:
        try:
            existing_metadata, _ = _open_regular_file(directory_fd, "metadata.json", None, "completion metadata")
        except FileNotFoundError:
            pass
        else:
            os.close(existing_metadata)
            raise FileExistsError(f"refusing to replace existing completion metadata: {resolved / 'metadata.json'}")
        for partial_name, final_name, expected_bytes, label in pairs:
            try:
                final_fd, final_identity = _open_regular_file(
                    directory_fd, final_name, expected_bytes, label
                )
            except FileNotFoundError:
                final_fd, final_identity = _publish_binary_no_replace(
                    directory_fd, partial_name, final_name, expected_bytes, label
                )
            else:
                try:
                    try:
                        partial_fd, partial_identity = _open_regular_file(
                            directory_fd, partial_name, expected_bytes, f"partial {label}"
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        os.close(partial_fd)
                        if partial_identity != final_identity:
                            raise RuntimeError(
                                f"cannot finalize {label}: partial and final files have different inodes"
                            )
                except BaseException:
                    os.close(final_fd)
                    raise
            held_finals.append((final_fd, final_identity, final_name, label))
        for _, identity, name, label in held_finals:
            _verify_named_identity(directory_fd, name, identity, label)
        metadata = _final_metadata(run, manifest, merge_state, batch_size, num_workers)
        metadata_identity, metadata_fd = _atomic_create_json(directory_fd, "metadata.json", metadata)
        try:
            for _, identity, name, label in held_finals:
                _verify_named_identity(directory_fd, name, identity, label)
        except BaseException:
            _invalidate_metadata_descriptor(metadata_fd)
            raise
        finally:
            os.close(metadata_fd)
        return metadata
    finally:
        for descriptor, _, _, _ in held_finals:
            os.close(descriptor)
        os.close(directory_fd)


def _completed_metadata_error() -> RuntimeError:
    return RuntimeError(
        "completed preprocessing output does not match this run; use a different output directory or explicit --overwrite"
    )


def load_completed_metadata(output_dir: Path, run: RunIdentity) -> dict[str, object] | None:
    resolved = output_dir.resolve(strict=False)
    directory_fd = _open_output_directory(resolved)
    try:
        try:
            metadata_fd, _ = _open_regular_file(directory_fd, "metadata.json", None, "completion metadata")
        except FileNotFoundError:
            return None
        try:
            with os.fdopen(os.dup(metadata_fd), "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        finally:
            os.close(metadata_fd)
        if not isinstance(payload, dict):
            raise ValueError("metadata is not an object")
        expected_keys = set(_final_metadata(run, RunManifest("placeholder", tuple()), MergeState(
            "placeholder", 0, TOKEN_BYTES, TOKEN_BYTES, 0, 0, 2
        ), 1, 1))
        if set(payload) != expected_keys:
            raise ValueError("metadata schema differs")
        int_keys = {
            "format_version", "source_files", "documents", "skipped_documents", "train_tokens",
            "validation_tokens", "total_tokens", "vocab_size", "num_workers", "batch_size",
        }
        str_keys = {
            "run_sha256", "manifest_sha256", "source_manifest_sha256", "dtype", "byte_order",
            "tokenizer_path", "tokenizer_sha256", "text_column",
        }
        if any(type(payload[key]) is not int for key in int_keys) or any(
            type(payload[key]) is not str for key in str_keys
        ) or type(payload["add_bos"]) is not bool or type(payload["add_eos"]) is not bool:
            raise ValueError("metadata types differ")
        if (
            payload["format_version"] != FORMAT_VERSION
            or payload["run_sha256"] != run.digest()
            or payload["tokenizer_sha256"] != run.tokenizer_sha256
            or payload["tokenizer_path"] != run.tokenizer_path
            or payload["vocab_size"] != run.vocab_size
            or payload["source_files"] != len(run.inputs)
            or payload["source_manifest_sha256"] != _source_manifest_sha256(run)
            or payload["add_bos"] != run.add_bos
            or payload["add_eos"] != run.add_eos
            or payload["text_column"] != run.text_column
            or payload["dtype"] != "int32"
            or payload["byte_order"] != "little"
            or payload["train_tokens"] <= 0
            or payload["validation_tokens"] < 1
            or payload["total_tokens"] != payload["train_tokens"] + payload["validation_tokens"]
        ):
            raise ValueError("metadata values differ")
        train_fd: int | None = None
        val_fd: int | None = None
        try:
            train_fd, _ = _open_regular_file(
                directory_fd, "train.bin", payload["train_tokens"] * TOKEN_BYTES, "training output"
            )
            val_fd, _ = _open_regular_file(
                directory_fd, "val.bin", payload["validation_tokens"] * TOKEN_BYTES, "validation output"
            )
        finally:
            if train_fd is not None:
                os.close(train_fd)
            if val_fd is not None:
                os.close(val_fd)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and str(error) == str(_completed_metadata_error()):
            raise
        raise _completed_metadata_error() from error
    finally:
        os.close(directory_fd)
    return payload


def print_disk_status(output_dir: Path, full_run: bool) -> None:
    usage = shutil.disk_usage(output_dir)
    free_gib = usage.free / (1024 ** 3)
    print(f"disk free at {output_dir}: {free_gib:.1f} GiB")
    if full_run and free_gib < 450:
        print(f"warning: only {free_gib:.1f} GiB free; a full run needs substantially more disk", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = prepare_output(Path(args.output_dir), args.overwrite, args.resume)
    inputs = discover_inputs(args.input_glob, args.max_files)
    tokenizer = SentencePieceTokenizer(Path(args.tokenizer).resolve())
    if tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"expected a {EXPECTED_VOCAB_SIZE}-piece tokenizer, got {tokenizer.vocab_size}")
    run = build_run_identity(
        inputs, Path(args.tokenizer), tokenizer.vocab_size, args.text_column, args.add_bos, args.add_eos,
    )
    print_disk_status(output_dir, full_run=args.max_files is None)
    if args.resume and load_completed_metadata(output_dir, run) is not None:
        print(f"preprocessing already complete: {output_dir}")
        return 0
    state_dir = output_dir / ".preprocess"
    tasks = make_worker_tasks(run, state_dir, args.batch_size)
    resume_context = load_resume_context(state_dir, run.digest()) if args.resume else None
    if resume_context is None:
        results = process_shards(tasks, run.tokenizer_path, args.num_workers, args.resume)
        manifest = write_run_manifest(state_dir, run.digest(), results)
        merge_state = merge_shards(results, output_dir, args.validation_tokens, manifest, False)
    else:
        manifest, previous_merge = resume_context
        if previous_merge.next_index == len(manifest.results):
            merge_state = previous_merge
        else:
            committed = list(manifest.results[:previous_merge.next_index])
            remaining = process_shards(
                tasks[previous_merge.next_index:], run.tokenizer_path, args.num_workers, True,
            )
            results = [*committed, *remaining]
            if tuple(results) != manifest.results:
                raise RuntimeError("rebuilt shard statistics differ from run manifest")
            merge_state = merge_shards(results, output_dir, args.validation_tokens, manifest, True)
    metadata = finalize_outputs(
        output_dir, run, manifest, merge_state, args.batch_size, args.num_workers, args.validation_tokens,
    )
    print(
        f"wrote {metadata['train_tokens']:,} training and "
        f"{metadata['validation_tokens']:,} validation tokens to {output_dir}"
    )
    return 0


def _interrupt_on_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    previous = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        raise SystemExit(main())
    finally:
        signal.signal(signal.SIGTERM, previous)
