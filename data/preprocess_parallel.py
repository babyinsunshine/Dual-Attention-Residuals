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
TOKEN_BYTES = np.dtype(TOKEN_DTYPE).itemsize
COPY_TOKENS = 16_777_216

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


def _manifest_payload(run_sha256: str, results: Sequence[ShardResult]) -> dict[str, object]:
    if type(run_sha256) is not str or not run_sha256:
        raise RuntimeError("run hash must be a non-empty string")
    normalized = _validate_results(results)
    return {"run_sha256": run_sha256, "results": [result.to_json() for result in normalized]}


def _atomic_write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if manifest_exists != state_exists:
        raise RuntimeError("resume requires both run-manifest.json and merge-state.json")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid run manifest: {manifest_path}") from error
    manifest = _manifest_from_payload(manifest_payload, run_sha256)
    return manifest, _load_merge_state(state_path, manifest)


def recover_output_file(path: Path, committed_bytes: int) -> None:
    if type(committed_bytes) is not int or committed_bytes < 0 or committed_bytes % TOKEN_BYTES:
        raise RuntimeError(f"invalid committed byte count for merge output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if committed_bytes:
            raise RuntimeError(f"missing committed merge output: {path}")
        path.touch()
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


def _state_dir_for_results(results: Sequence[ShardResult], output_dir: Path) -> Path:
    if not results:
        return output_dir / ".preprocess"
    state_dir = Path(results[0].shard_path).parent.parent
    if any(Path(result.shard_path).parent.parent != state_dir for result in results):
        raise RuntimeError("all merge shards must share one state directory")
    return state_dir


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
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = _state_dir_for_results(normalized, output_dir)
    state_path = state_dir / "merge-state.json"
    state = _load_merge_state(state_path, run_manifest) if resume and state_path.exists() else _initial_merge_state(run_manifest)
    committed_tokens = sum(item.tokens for item in normalized[:state.next_index])
    expected_validation_bytes = min(committed_tokens, validation_tokens) * TOKEN_BYTES
    if state.validation_bytes != expected_validation_bytes:
        raise RuntimeError("merge state validation bytes do not match the requested split")
    validation_path = output_dir / "val.bin.partial"
    training_path = output_dir / "train.bin.partial"
    recover_output_file(validation_path, state.validation_bytes)
    recover_output_file(training_path, state.training_bytes)
    for committed in normalized[:state.next_index]:
        Path(committed.shard_path).unlink(missing_ok=True)
        Path(committed.sidecar_path).unlink(missing_ok=True)
    with validation_path.open("ab") as validation_output, training_path.open("ab") as training_output:
        for position, result in enumerate(normalized[state.next_index:], start=state.next_index):
            shard = Path(result.shard_path)
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
            Path(result.sidecar_path).unlink()
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
