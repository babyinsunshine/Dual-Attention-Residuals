# Parallel FineWeb-Edu 100BT Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, resumable 64-process converter from local FineWeb-Edu Parquet shards to the single `train.bin` and `val.bin` token files consumed by this repository.

**Architecture:** `data/preprocess_parallel.py` owns input discovery, worker initialization, per-Parquet token shards, process-pool orchestration, rolling merge recovery, metadata, and the CLI. Workers stream only the `text` column through PyArrow and write atomic little-endian `int32` shards; the parent merges in lexical source order and uses an atomic journal so completed shards can be deleted without sacrificing recovery.

**Tech Stack:** Python 3.10, standard-library `argparse`/`concurrent.futures`/`multiprocessing`/`unittest`, NumPy 1.26.4, PyArrow 21.0.0, SentencePiece 0.2.1.

## Global Constraints

- Keep `data/preprocess.py` and the training loader behavior unchanged.
- Input Parquet files are read-only and must never be renamed, modified, or deleted.
- Default workers are `min(64, os.cpu_count() or 1)`; PyArrow internal threading is disabled in every worker.
- Inputs are ordered by normalized absolute path; output bytes must not depend on worker completion order.
- Token storage is explicit little-endian signed `int32` (`<i4`).
- `val.bin` contains exactly 10,000,000 tokens by default; `train.bin` must be non-empty.
- Final files are not overwritten without `--overwrite`; deletion is limited to named files under the resolved output directory.
- `metadata.json` is written last and is the successful-completion marker.
- Declare `pyarrow==21.0.0` directly in both `requirements.txt` and `pyproject.toml`.
- Do not stage or modify the user's existing `dual_attentions_residuals/mcore_compat.py` change.

---

## File structure

- Create `data/preprocess_parallel.py`: complete preprocessing library and CLI; functions stay importable for unit tests.
- Create `tests/__init__.py`: make targeted standard-library test invocation reliable.
- Create `tests/test_preprocess_parallel.py`: synthetic Parquet, resume, merge journal, and safety tests.
- Modify `requirements.txt`: pin direct PyArrow dependency.
- Modify `pyproject.toml`: declare the same runtime dependency.
- Modify `README.md`: document local parallel preprocessing and 100BT disk requirements.

### Task 1: Input discovery, run identity, and safe CLI validation

**Files:**
- Create: `data/preprocess_parallel.py`
- Create: `tests/__init__.py`
- Create: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Produces: `parse_args(argv: Sequence[str] | None) -> argparse.Namespace`
- Produces: `discover_inputs(pattern: str, max_files: int | None) -> list[Path]`
- Produces: `FileIdentity.from_path(index: int, path: Path) -> FileIdentity`
- Produces: `RunIdentity` and `build_run_identity(inputs, tokenizer_path, vocab_size, text_column, add_bos, add_eos) -> RunIdentity`
- Produces: `atomic_write_json(path: Path, payload: Mapping[str, object]) -> None`
- Consumes: `data.tokenizer.SentencePieceTokenizer` and the tokenizer SHA-256 convention from `data/preprocess.py`

- [ ] **Step 1: Write failing discovery and validation tests**

Add `tests/__init__.py` as an empty file. Add these tests to
`tests/test_preprocess_parallel.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data import preprocess_parallel as pp


class DiscoveryTests(unittest.TestCase):
    def test_discover_inputs_sorts_resolved_paths_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("b.parquet", "a.parquet", "c.txt"):
                (root / name).touch()
            found = pp.discover_inputs(str(root / "*.parquet"), max_files=1)
            self.assertEqual(found, [(root / "a.parquet").resolve()])

    def test_discover_inputs_rejects_empty_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "matched no Parquet files"):
                pp.discover_inputs(str(Path(tmp) / "*.parquet"), max_files=None)

    def test_parse_args_rejects_nonpositive_values(self) -> None:
        with self.assertRaises(SystemExit):
            pp.parse_args(["--input-glob", "x", "--num-workers", "0"])
        with self.assertRaises(SystemExit):
            pp.parse_args(["--input-glob", "x", "--validation-tokens", "0"])

    def test_atomic_write_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            pp.atomic_write_json(path, {"value": 1})
            pp.atomic_write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 2})
            self.assertFalse(path.with_suffix(".json.partial").exists())
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
python -m unittest tests.test_preprocess_parallel.DiscoveryTests -v
```

Expected: `ImportError` because `data.preprocess_parallel` does not exist.

- [ ] **Step 3: Implement the parser, identities, hashing, and atomic JSON helper**

Create `data/preprocess_parallel.py` with these exact public shapes and validations:

```python
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
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
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


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
```

- [ ] **Step 4: Run the discovery tests**

```bash
python -m unittest tests.test_preprocess_parallel.DiscoveryTests -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit the input and identity foundation**

```bash
git add data/preprocess_parallel.py tests/__init__.py tests/test_preprocess_parallel.py
git commit -m "feat: add parallel preprocessing input model"
```

### Task 2: Atomic token-shard worker and shard resume validation

**Files:**
- Modify: `data/preprocess_parallel.py`
- Modify: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Consumes: `FileIdentity`, `RunIdentity`, `TOKEN_DTYPE`, `atomic_write_json`
- Produces: `WorkerTask`, `ShardResult`, `initialize_worker`, `tokenize_file`, `load_valid_shard`, `make_worker_tasks`

- [ ] **Step 1: Write failing worker and resume tests**

Append these test fixtures and cases:

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class FakeTokenizer:
    vocab_size = 131_072

    def encode_batch(self, texts, *, add_bos=False, add_eos=False):
        rows = []
        for text in texts:
            ids = [len(text), sum(text.encode("utf-8"))]
            if add_bos:
                ids.insert(0, 1)
            if add_eos:
                ids.append(2)
            rows.append(ids)
        return rows


class WorkerTests(unittest.TestCase):
    def make_task(self, root: Path) -> pp.WorkerTask:
        source = root / "000.parquet"
        pq.write_table(pa.table({"text": ["a", None, "  ", "bc"]}), source)
        return pp.WorkerTask(
            source=pp.FileIdentity.from_path(0, source),
            shard_path=str(root / "state" / "000000.bin"),
            sidecar_path=str(root / "state" / "000000.json"),
            tokenizer_sha256="tokenizer-hash",
            text_column="text", batch_size=2, add_bos=False, add_eos=True,
        )

    def test_tokenize_file_writes_atomic_little_endian_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = FakeTokenizer()
            result = pp.tokenize_file(task)
            self.assertEqual(np.fromfile(result.shard_path, dtype="<i4").tolist(), [1, 97, 2, 2, 197, 2])
            self.assertEqual((result.documents, result.skipped_documents, result.tokens), (2, 2, 6))
            self.assertFalse(Path(result.shard_path + ".partial").exists())

    def test_load_valid_shard_reuses_only_matching_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = FakeTokenizer()
            result = pp.tokenize_file(task)
            self.assertEqual(pp.load_valid_shard(task), result)
            Path(result.shard_path).write_bytes(b"broken")
            self.assertIsNone(pp.load_valid_shard(task))

    def test_tokenize_file_rejects_token_ids_outside_int32(self) -> None:
        class OverflowTokenizer(FakeTokenizer):
            def encode_batch(self, texts, *, add_bos=False, add_eos=False):
                return [[2**31] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = OverflowTokenizer()
            with self.assertRaisesRegex(OverflowError, "int32"):
                pp.tokenize_file(task)
```

- [ ] **Step 2: Run worker tests and verify missing interfaces**

```bash
python -m unittest tests.test_preprocess_parallel.WorkerTests -v
```

Expected: errors naming missing `WorkerTask` and `tokenize_file`.

- [ ] **Step 3: Implement worker records and batched tokenization**

Add `numpy`, `pyarrow`, `pyarrow.parquet`, `SentencePieceTokenizer`, and
`TokenizerLike` imports. Define the records and worker initializer:

```python
_WORKER_TOKENIZER: TokenizerLike | None = None


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
```

Implement `load_valid_shard` by requiring both files, exact serialized task
identity, and `shard_size == tokens * 4`. Implement `tokenize_file` with this
ordered flow:

```python
def tokenize_file(task: WorkerTask) -> ShardResult:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    shard, sidecar = Path(task.shard_path), Path(task.sidecar_path)
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
        partial.replace(shard)
        result = ShardResult(task.source, str(shard), str(sidecar), documents, skipped, tokens)
        atomic_write_json(sidecar, {"task": asdict(task), "result": result.to_json()})
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
```

`make_worker_tasks(run, state_dir, batch_size)` produces one task per input with
zero-padded names `shards/000000.bin` and `shards/000000.json`.

- [ ] **Step 4: Run all current tests**

```bash
python -m unittest tests.test_preprocess_parallel -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit shard tokenization**

```bash
git add data/preprocess_parallel.py tests/test_preprocess_parallel.py
git commit -m "feat: tokenize Parquet files into resumable shards"
```

### Task 3: Process-pool scheduling and visible progress

**Files:**
- Modify: `data/preprocess_parallel.py`
- Modify: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Consumes: `WorkerTask`, `ShardResult`, `initialize_worker`, `load_valid_shard`, `tokenize_file`
- Produces: `summarize_progress(done, total, started_at, now=None) -> str`
- Produces: `_run_pending_tasks(tasks, tokenizer_path, num_workers, already_done, total, started_at, heartbeat_seconds, emit) -> list[ShardResult]`
- Produces: `process_shards(tasks, tokenizer_path, num_workers, resume, heartbeat_seconds=30.0, emit=print) -> list[ShardResult]`

- [ ] **Step 1: Write failing orchestration tests**

Use `unittest.mock.patch` so this unit test stays synchronous:

```python
from unittest import mock


class ProgressTests(unittest.TestCase):
    def test_process_shards_preserves_source_order_and_reports_resume(self) -> None:
        tasks = [
            pp.WorkerTask(
                pp.FileIdentity(i, f"/{name}", 10, 20), f"/{i}.bin", f"/{i}.json",
                "hash", "text", 2, False, True,
            )
            for i, name in enumerate(("a.parquet", "b.parquet"))
        ]
        reused = pp.ShardResult(tasks[0].source, "/0.bin", "/0.json", 2, 0, 8)
        built = pp.ShardResult(tasks[1].source, "/1.bin", "/1.json", 3, 1, 12)
        messages = []
        with mock.patch.object(pp, "load_valid_shard", side_effect=[reused, None]), mock.patch.object(
            pp, "_run_pending_tasks", return_value=[built]
        ):
            results = pp.process_shards(tasks, "tokenizer.model", 4, True, emit=messages.append)
        self.assertEqual([item.source.index for item in results], [0, 1])
        self.assertTrue(any("reused=1" in line for line in messages))

    def test_summarize_progress_contains_counts_and_rate(self) -> None:
        result = pp.ShardResult(pp.FileIdentity(0, "/a", 1, 1), "/a.bin", "/a.json", 4, 1, 100)
        message = pp.summarize_progress([result], total=2, started_at=10.0, now=20.0)
        self.assertIn("files=1/2", message)
        self.assertIn("tokens=100", message)
        self.assertIn("tokens/s=10.0", message)
```

- [ ] **Step 2: Run progress tests and verify missing functions**

```bash
python -m unittest tests.test_preprocess_parallel.ProgressTests -v
```

Expected: errors naming `process_shards` and `summarize_progress`.

- [ ] **Step 3: Implement spawn-based scheduling and heartbeat**

Import `time`, `FIRST_COMPLETED`, `ProcessPoolExecutor`, `wait`,
`multiprocessing.get_context`, and `Callable`. Implement:

```python
def summarize_progress(done, total, started_at, now=None) -> str:
    current = time.monotonic() if now is None else now
    elapsed = max(current - started_at, 1e-9)
    documents = sum(item.documents for item in done)
    tokens = sum(item.tokens for item in done)
    return (
        f"files={len(done)}/{total} documents={documents:,} tokens={tokens:,} "
        f"elapsed={elapsed:.1f}s tokens/s={tokens / elapsed:,.1f}"
    )
```

`_run_pending_tasks` creates a `ProcessPoolExecutor` with
`mp_context=get_context("spawn")`, `initializer=initialize_worker`, and the
tokenizer path in `initargs`. Submit one `tokenize_file` per pending task. Call
`wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)` in a loop; emit a
heartbeat on timeout and after each completion. On an exception, cancel all
pending futures and raise `RuntimeError(f"failed to preprocess
{task.source.path}")` chained from the worker exception.

`process_shards` keeps separate typed `reused: list[ShardResult]` and
`pending: list[WorkerTask]` lists, prints `files=N reused=R pending=P`, calls the
executor only when pending is non-empty, sorts all results by source index, and
requires the result indexes to equal the input task indexes. This permits a
resume call whose first uncommitted task index is greater than zero.

- [ ] **Step 4: Run all current tests**

```bash
python -m unittest tests.test_preprocess_parallel -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit parallel scheduling**

```bash
git add data/preprocess_parallel.py tests/test_preprocess_parallel.py
git commit -m "feat: process token shards in parallel"
```

### Task 4: Journaled rolling merge and exact validation split

**Files:**
- Modify: `data/preprocess_parallel.py`
- Modify: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Consumes: ordered `Sequence[ShardResult]`, run digest, and `TOKEN_DTYPE`
- Produces: `RunManifest`, `MergeState`, `write_run_manifest(state_dir, run_sha256, results) -> RunManifest`
- Produces: `load_resume_context(state_dir, run_sha256) -> tuple[RunManifest, MergeState] | None`
- Produces: `merge_shards(results, output_dir, validation_tokens, run_manifest, resume) -> MergeState`
- Produces: `recover_output_file(path, committed_bytes) -> None`

- [ ] **Step 1: Write failing exact-split and interrupted-resume tests**

Create known token shards and verify exact splitting, ordered bytes, deletion
after commit, and truncation of uncommitted bytes:

```python
class MergeTests(unittest.TestCase):
    def write_result(self, root: Path, index: int, values: list[int]) -> pp.ShardResult:
        shard = root / ".preprocess" / "shards" / f"{index:06d}.bin"
        sidecar = shard.with_suffix(".json")
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(values, dtype="<i4").tofile(shard)
        sidecar.write_text("{}")
        return pp.ShardResult(
            pp.FileIdentity(index, f"/source/{index}.parquet", 1, 1),
            str(shard), str(sidecar), 1, 0, len(values),
        )

    def test_merge_splits_exact_validation_and_deletes_committed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [self.write_result(root, 0, [1, 2, 3]), self.write_result(root, 1, [4, 5, 6])]
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", results)
            state = pp.merge_shards(results, root, 4, manifest, resume=False)
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [1, 2, 3, 4])
            self.assertEqual(np.fromfile(root / "train.bin.partial", dtype="<i4").tolist(), [5, 6])
            self.assertEqual(state.next_index, 2)
            self.assertFalse(Path(results[0].shard_path).exists())

    def test_merge_resume_truncates_uncommitted_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [7, 8, 9])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            val = root / "val.bin.partial"
            np.asarray([7, 99], dtype="<i4").tofile(val)
            pp.atomic_write_json(root / ".preprocess" / "merge-state.json", {
                "manifest_sha256": manifest.sha256,
                "next_index": 0,
                "validation_bytes": 4,
                "training_bytes": 0,
                "documents": 0,
                "skipped_documents": 0,
                "tokens": 0,
            })
            pp.merge_shards([result], root, 3, manifest, resume=True)
            self.assertEqual(np.fromfile(val, dtype="<i4").tolist(), [7, 8, 9])

    def test_resume_context_does_not_require_deleted_committed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [self.write_result(root, 0, [1]), self.write_result(root, 1, [2])]
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", results)
            pp.atomic_write_json(root / ".preprocess" / "merge-state.json", {
                "manifest_sha256": manifest.sha256,
                "next_index": 1,
                "validation_bytes": 4,
                "training_bytes": 0,
                "documents": 1,
                "skipped_documents": 0,
                "tokens": 1,
            })
            Path(results[0].shard_path).unlink()
            Path(results[0].sidecar_path).unlink()
            loaded_manifest, loaded_state = pp.load_resume_context(
                root / ".preprocess", "run-hash"
            )
            self.assertEqual(loaded_manifest.results, tuple(results))
            self.assertEqual(loaded_state.next_index, 1)
```

- [ ] **Step 2: Run merge tests and verify missing interfaces**

```bash
python -m unittest tests.test_preprocess_parallel.MergeTests -v
```

Expected: errors naming `write_run_manifest` and `merge_shards`.

- [ ] **Step 3: Implement immutable run manifest and merge journal records**

Define immutable records:

```python
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
```

`write_run_manifest` serializes `run_sha256` plus ordered
`ShardResult.to_json()` records using canonical JSON, hashes those exact bytes,
writes `run-manifest.json` atomically, and returns `RunManifest`. If a manifest
already exists, require identical canonical content instead of silently
replacing a different run.

`load_resume_context` returns `None` when neither state file exists. A valid
manifest without state atomically establishes the zero merge state; state without
a manifest is a hard error. It is also a hard error when the stored run hash
differs, when the merge manifest hash differs, or when `next_index` is outside
`0..len(manifest.results)`. This function is called before worker scheduling so
already committed and deleted shards are never rebuilt.

Implement exact file recovery:

```python
def recover_output_file(path: Path, committed_bytes: int) -> None:
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
```

- [ ] **Step 4: Implement chunked, journal-before-delete rolling merge**

`merge_shards` must perform this exact order for each lexical shard:

1. Load merge state only under `resume=True`, verify manifest hash, then recover
   both partial outputs to journaled sizes.
2. Verify `shard.stat().st_size == result.tokens * 4` and memory-map as `<i4`.
3. Copy chunks of at most 16,777,216 tokens (64 MiB), first filling the remaining
   validation quota and then appending to training.
4. Flush and `fsync` both destination streams.
5. Atomically write `merge-state.json` with incremented index, committed byte
   lengths, and cumulative document/token counts.
6. Only after step 5 succeeds, unlink that exact shard and sidecar.

Use this bounded copy helper rather than reading a full shard:

```python
COPY_TOKENS = 16_777_216


def write_token_range(source, start: int, stop: int, destination) -> int:
    written = 0
    for offset in range(start, stop, COPY_TOKENS):
        end = min(stop, offset + COPY_TOKENS)
        chunk = np.asarray(source[offset:end], dtype=TOKEN_DTYPE)
        payload = chunk.tobytes(order="C")
        destination.write(payload)
        written += len(payload)
    return written
```

Return the final `MergeState`; leave outputs named `train.bin.partial` and
`val.bin.partial` for safe finalization.

- [ ] **Step 5: Run all tests**

```bash
python -m unittest tests.test_preprocess_parallel -v
```

Expected: all 12 tests pass.

- [ ] **Step 6: Commit rolling merge recovery**

```bash
git add data/preprocess_parallel.py tests/test_preprocess_parallel.py
git commit -m "feat: add resumable token shard merge"
```

### Task 5: End-to-end CLI, metadata, overwrite safety, and interruption recovery

**Files:**
- Modify: `data/preprocess_parallel.py`
- Modify: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Consumes: all previous task interfaces
- Produces: `prepare_output(output_dir, overwrite, resume) -> Path`
- Produces: `finalize_outputs(output_dir, run, manifest, merge_state, batch_size, num_workers, validation_tokens) -> dict[str, object]`
- Produces: `load_completed_metadata(output_dir, run) -> dict[str, object] | None`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing safety and finalization tests**

Add these cases and a third case where binaries have already been renamed but
metadata is absent:

```python
class FinalizationTests(unittest.TestCase):
    def test_existing_final_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.bin").write_bytes(b"user-data")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                pp.prepare_output(root, overwrite=False, resume=False)
            self.assertEqual((root / "train.bin").read_bytes(), b"user-data")

    def test_finalize_writes_metadata_last_and_renames_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            run = pp.RunIdentity(1, tuple(), "/tokenizer.model", "abc", 131072, "text", False, True)
            manifest = pp.RunManifest("manifest", tuple())
            state = pp.MergeState("manifest", 0, 8, 4, 1, 0, 3)
            metadata = pp.finalize_outputs(
                root, run, manifest, state, batch_size=2,
                num_workers=1, validation_tokens=2,
            )
            self.assertTrue((root / "train.bin").is_file())
            self.assertTrue((root / "val.bin").is_file())
            self.assertEqual(json.loads((root / "metadata.json").read_text()), metadata)

    def test_finalize_recovers_when_binaries_were_already_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin")
            run = pp.RunIdentity(1, tuple(), "/tokenizer.model", "abc", 131072, "text", False, True)
            manifest = pp.RunManifest("manifest", tuple())
            state = pp.MergeState("manifest", 0, 8, 4, 1, 0, 3)
            pp.finalize_outputs(
                root, run, manifest, state, batch_size=2,
                num_workers=1, validation_tokens=2,
            )
            self.assertTrue((root / "metadata.json").is_file())
            self.assertEqual(np.fromfile(root / "val.bin", dtype="<i4").tolist(), [1, 2])
            self.assertEqual(np.fromfile(root / "train.bin", dtype="<i4").tolist(), [3])


class EndToEndTests(unittest.TestCase):
    def test_one_and_two_workers_produce_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            pq.write_table(pa.table({"text": ["alpha beta", "gamma delta"]}), inputs / "b.parquet")
            pq.write_table(pa.table({"text": ["epsilon zeta", "eta theta"]}), inputs / "a.parquet")
            tokenizer = Path(pp.__file__).resolve().parents[1] / "tokenizer" / "tokenizer.model"
            outputs = []
            for workers in (1, 2):
                output = root / f"out-{workers}"
                exit_code = pp.main([
                    "--input-glob", str(inputs / "*.parquet"),
                    "--tokenizer", str(tokenizer),
                    "--output-dir", str(output),
                    "--num-workers", str(workers),
                    "--batch-size", "1",
                    "--validation-tokens", "5",
                    "--add-eos",
                ])
                self.assertEqual(exit_code, 0)
                outputs.append(output)
            self.assertEqual((outputs[0] / "val.bin").read_bytes(), (outputs[1] / "val.bin").read_bytes())
            self.assertEqual((outputs[0] / "train.bin").read_bytes(), (outputs[1] / "train.bin").read_bytes())
```

- [ ] **Step 2: Run finalization tests and verify missing functions**

```bash
python -m unittest tests.test_preprocess_parallel.FinalizationTests -v
```

Expected: errors naming `prepare_output` and `finalize_outputs`.

- [ ] **Step 3: Implement named-path cleanup and completion recovery**

Use this explicit owned-file list:

```python
OWNED_FINAL_NAMES = (
    "train.bin", "val.bin", "metadata.json",
    "train.bin.partial", "val.bin.partial",
)
```

`prepare_output` resolves and creates the output directory. If any final exists
and neither resume nor overwrite is set, raise `FileExistsError` mentioning both
flags. Under overwrite, unlink only `OWNED_FINAL_NAMES`; validate
`state_dir.parent == output_dir` and `state_dir.name == ".preprocess"` before
recursively removing state files and now-empty state directories. Never follow
symlinks during cleanup.

`finalize_outputs` must:

1. Require merge state token bytes to be divisible by four.
2. Require exactly the requested validation tokens and positive training tokens.
3. Rename each partial binary only if its final path is absent; if already
   renamed, validate its size against merge state.
4. Build metadata with exactly these keys: `format_version`, `run_sha256`,
   `manifest_sha256`, `source_files`, `source_manifest_sha256`, `documents`,
   `skipped_documents`, `train_tokens`, `validation_tokens`, `total_tokens`,
   `dtype`, `byte_order`, `tokenizer_path`, `tokenizer_sha256`, `vocab_size`,
   `add_bos`, `add_eos`, `text_column`, `num_workers`, and `batch_size`.
5. Atomically write `metadata.json` last.

`load_completed_metadata` accepts a no-op resume only when metadata run hash,
both final byte sizes, and tokenizer hash match the current `RunIdentity`. If
metadata exists but any check differs, raise `RuntimeError` instructing the
operator to use a different output directory or explicit `--overwrite`; never
return `None` and append to mismatched final files.

- [ ] **Step 4: Wire the CLI without destructive signal cleanup**

Implement `main` in this order:

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = prepare_output(Path(args.output_dir), args.overwrite, args.resume)
    inputs = discover_inputs(args.input_glob, args.max_files)
    tokenizer = SentencePieceTokenizer(Path(args.tokenizer).resolve())
    if tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"expected a {EXPECTED_VOCAB_SIZE}-piece tokenizer, got {tokenizer.vocab_size}"
        )
    run = build_run_identity(
        inputs, Path(args.tokenizer), tokenizer.vocab_size,
        args.text_column, args.add_bos, args.add_eos,
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
    else:
        manifest, previous_merge = resume_context
        committed = list(manifest.results[:previous_merge.next_index])
        remaining = process_shards(
            tasks[previous_merge.next_index:], run.tokenizer_path,
            args.num_workers, True,
        )
        results = [*committed, *remaining]
        if tuple(results) != manifest.results:
            raise RuntimeError("rebuilt shard statistics differ from run manifest")
    merge_state = merge_shards(results, output_dir, args.validation_tokens, manifest, args.resume)
    metadata = finalize_outputs(
        output_dir, run, manifest, merge_state,
        args.batch_size, args.num_workers, args.validation_tokens,
    )
    print(
        f"wrote {metadata['train_tokens']:,} training and "
        f"{metadata['validation_tokens']:,} validation tokens to {output_dir}"
    )
    return 0
```

`print_disk_status` uses `shutil.disk_usage`, prints free GiB, and warns on
stderr below 450 GiB only for a full run. In `__main__`, temporarily map
`SIGTERM` to a handler that raises `KeyboardInterrupt`, restore the previous
handler in `finally`. On `BaseException`, orchestration cancels futures,
requests non-waiting executor shutdown, and terminates spawned workers via the
contained Python-3.10 process-map compatibility helper before joining. The
handler must not delete files; atomic files and the journal provide recovery:

```python
def _interrupt_on_sigterm(signum, frame) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    previous = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        raise SystemExit(main())
    finally:
        signal.signal(signal.SIGTERM, previous)
```

- [ ] **Step 5: Run all focused tests**

```bash
python -m unittest tests.test_preprocess_parallel -v
```

Expected: all tests pass, including existing-output refusal, missing-metadata
recovery, and metadata-last completion.

- [ ] **Step 6: Commit the end-to-end CLI**

```bash
git add data/preprocess_parallel.py tests/test_preprocess_parallel.py
git commit -m "feat: finalize parallel preprocessing CLI"
```

### Task 6: Dependencies, README workflow, and local verification

**Files:**
- Modify: `requirements.txt:5-9`
- Modify: `pyproject.toml:12-21`
- Modify: `README.md:29-56`
- Test: `tests/test_preprocess_parallel.py`

**Interfaces:**
- Consumes: completed `data/preprocess_parallel.py` CLI
- Produces: reproducible dependency declaration and copy-pasteable 100BT usage

- [ ] **Step 1: Add the direct PyArrow pin**

Insert after NumPy in `requirements.txt`:

```text
pyarrow==21.0.0
```

Insert after NumPy in `pyproject.toml`:

```toml
  "pyarrow==21.0.0",
```

- [ ] **Step 2: Install only CPU preprocessing dependencies and run tests**

Avoid the unrelated Transformer Engine wheel while checking preprocessing:

```bash
uv pip install numpy==1.26.4 pyarrow==21.0.0 sentencepiece==0.2.1
python -m unittest tests.test_preprocess_parallel -v
```

Expected: all tests pass.

- [ ] **Step 3: Document smoke and full local-Parquet commands**

After the existing streaming example in `README.md`, add a local 100BT section
with these exact commands:

```bash
python data/preprocess_parallel.py \
  --input-glob '/path/to/fineweb-100B/sample/100BT/*.parquet' \
  --output-dir data/processed_smoke \
  --num-workers 2 \
  --max-files 2 \
  --validation-tokens 100000 \
  --add-eos

python data/preprocess_parallel.py \
  --input-glob '/path/to/fineweb-100B/sample/100BT/*.parquet' \
  --output-dir data/processed \
  --num-workers 64 \
  --validation-tokens 10000000 \
  --add-eos \
  --resume
```

State that output is approximately 400 GB, at least 450 GiB free is
recommended, progress appears every 30 seconds, rerunning with identical
arguments and `--resume` continues safely, and raw Parquet is never changed.

- [ ] **Step 4: Run static and CLI checks**

```bash
python -m compileall -q data/preprocess_parallel.py tests/test_preprocess_parallel.py
python data/preprocess_parallel.py --help
git diff --check
```

Expected: compilation exits 0, help lists `--input-glob`, `--num-workers`,
`--resume`, `--max-files`, and `--overwrite`, and diff check is silent.

- [ ] **Step 5: Run a two-file real-data smoke test on the target server**

```bash
python data/preprocess_parallel.py \
  --input-glob '/gemini/space/private/liyining/datasets/fineweb-100B/sample/100BT/*.parquet' \
  --tokenizer tokenizer/tokenizer.model \
  --output-dir data/processed_smoke \
  --num-workers 2 \
  --max-files 2 \
  --validation-tokens 100000 \
  --add-eos
```

Expected: progress is printed, metadata reports exactly 100,000 validation
tokens and positive training tokens, and both binaries open with
`np.memmap(path, dtype=np.int32, mode="r")`.

- [ ] **Step 6: Commit dependency and usage documentation**

```bash
git add requirements.txt pyproject.toml README.md
git commit -m "docs: document parallel FineWeb preprocessing"
```

### Task 7: Final regression and scope review

**Files:**
- Verify only; no planned modifications

**Interfaces:**
- Consumes: all prior deliverables
- Produces: completion evidence without including unrelated user work

- [ ] **Step 1: Run the complete preprocessing test suite**

```bash
python -m unittest tests.test_preprocess_parallel -v
```

Expected: every test passes.

- [ ] **Step 2: Re-run compilation and whitespace validation**

```bash
python -m compileall -q data/preprocess_parallel.py tests/test_preprocess_parallel.py
git diff --check HEAD~5..HEAD
```

Expected: both commands exit 0 and diff check is silent.

- [ ] **Step 3: Confirm commit scope and preserve user changes**

```bash
git status --short
git log --oneline -6
git diff -- dual_attentions_residuals/mcore_compat.py
```

Expected: the pre-existing `dual_attentions_residuals/mcore_compat.py` change is
still unstaged and absent from feature commits; only planned feature files
appear in implementation commits.

- [ ] **Step 4: Record target-server smoke evidence**

Capture the exact command, elapsed time, source files, token throughput,
`metadata.json`, and output byte sizes in the implementation handoff. Do not
claim the full 100BT run is verified unless that run actually completes.
