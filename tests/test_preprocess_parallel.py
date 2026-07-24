from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

    def test_parse_args_uses_worker_default_and_override(self) -> None:
        default = pp.parse_args(["--input-glob", "x"])
        override = pp.parse_args(["--input-glob", "x", "--num-workers", "3"])
        self.assertEqual(default.num_workers, min(64, os.cpu_count() or 1))
        self.assertEqual(override.num_workers, 3)

    def test_parse_args_rejects_resume_and_overwrite_together(self) -> None:
        with self.assertRaises(SystemExit):
            pp.parse_args(["--input-glob", "x", "--resume", "--overwrite"])

    def test_atomic_write_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            pp.atomic_write_json(path, {"value": 1})
            pp.atomic_write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 2})
            self.assertFalse(path.with_suffix(".json.partial").exists())

    def test_atomic_write_json_cleans_up_temporary_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TypeError):
                pp.atomic_write_json(root / "state.json", {"value": object()})
            self.assertEqual(list(root.iterdir()), [])

    def test_atomic_write_json_allows_overlapping_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            barrier = threading.Barrier(2)
            errors: list[Exception] = []
            original_replace = os.replace

            def coordinated_replace(source: str | Path, target: str | Path) -> None:
                barrier.wait(timeout=5)
                return original_replace(source, target)

            def write(value: int) -> None:
                try:
                    pp.atomic_write_json(path, {"value": value})
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)

            with mock.patch.object(pp.os, "replace", coordinated_replace):
                writers = [threading.Thread(target=write, args=(value,)) for value in (1, 2)]
                for writer in writers:
                    writer.start()
                for writer in writers:
                    writer.join()

            self.assertEqual(errors, [])
            self.assertIn(json.loads(path.read_text())["value"], (1, 2))

    def test_build_run_identity_is_deterministic_and_resolves_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.parquet"
            tokenizer_path = root / "tokenizer.model"
            input_path.write_bytes(b"input")
            tokenizer_path.write_bytes(b"tokenizer")
            first = pp.build_run_identity(
                [input_path], tokenizer_path, pp.EXPECTED_VOCAB_SIZE, "text", True, False
            )
            second = pp.build_run_identity(
                [input_path], tokenizer_path, pp.EXPECTED_VOCAB_SIZE, "text", True, False
            )
            self.assertEqual(first.digest(), second.digest())
            self.assertEqual(first.tokenizer_path, str(tokenizer_path.resolve()))
            self.assertEqual(first.tokenizer_sha256, pp.sha256(tokenizer_path))


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

    def test_load_valid_shard_rejects_nonexact_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = FakeTokenizer()
            result = pp.tokenize_file(task)
            payload = json.loads(Path(result.sidecar_path).read_text())
            payload["result"]["unexpected"] = "value"
            Path(result.sidecar_path).write_text(json.dumps(payload))
            self.assertIsNone(pp.load_valid_shard(task))

    def test_load_valid_shard_rejects_numeric_and_boolean_type_substitutions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = FakeTokenizer()
            result = pp.tokenize_file(task)
            original = Path(result.sidecar_path).read_text()
            substitutions = (
                ("task", "batch_size", 2.0),
                ("task", "add_eos", 1),
                ("result", "tokens", 6.0),
                ("result", "documents", True),
            )
            for section, field, value in substitutions:
                payload = json.loads(original)
                payload[section][field] = value
                Path(result.sidecar_path).write_text(json.dumps(payload))
                self.assertIsNone(pp.load_valid_shard(task), (section, field, value))
            payload = json.loads(original)
            payload["result"]["source"]["index"] = False
            Path(result.sidecar_path).write_text(json.dumps(payload))
            self.assertIsNone(pp.load_valid_shard(task), "result.source.index")

    def test_tokenize_file_rejects_token_ids_outside_int32(self) -> None:
        class OverflowTokenizer(FakeTokenizer):
            def encode_batch(self, texts, *, add_bos=False, add_eos=False):
                return [[2**31] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = OverflowTokenizer()
            with self.assertRaisesRegex(OverflowError, "int32"):
                pp.tokenize_file(task)

    def test_tokenize_file_accepts_lowest_int32_token_id(self) -> None:
        class MinimumTokenizer(FakeTokenizer):
            def encode_batch(self, texts, *, add_bos=False, add_eos=False):
                return [[-(2**31)] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = MinimumTokenizer()
            result = pp.tokenize_file(task)
            self.assertEqual(np.fromfile(result.shard_path, dtype="<i4").tolist(), [-(2**31), -(2**31)])

    def test_tokenize_file_rejects_token_ids_below_int32(self) -> None:
        class UnderflowTokenizer(FakeTokenizer):
            def encode_batch(self, texts, *, add_bos=False, add_eos=False):
                return [[-(2**31) - 1] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            task = self.make_task(Path(tmp))
            pp._WORKER_TOKENIZER = UnderflowTokenizer()
            with self.assertRaisesRegex(OverflowError, "int32"):
                pp.tokenize_file(task)


class ProgressTests(unittest.TestCase):
    @staticmethod
    def make_tasks(*indexes: int) -> list[pp.WorkerTask]:
        return [
            pp.WorkerTask(
                pp.FileIdentity(index, f"/{index}.parquet", 10, 20),
                f"/{index}.bin",
                f"/{index}.json",
                "hash",
                "text",
                2,
                False,
                True,
            )
            for index in indexes
        ]

    @staticmethod
    def make_result(task: pp.WorkerTask, tokens: int = 8) -> pp.ShardResult:
        return pp.ShardResult(task.source, task.shard_path, task.sidecar_path, 2, 0, tokens)

    def test_process_shards_preserves_source_order_and_reports_resume(self) -> None:
        tasks = self.make_tasks(0, 1)
        reused = self.make_result(tasks[0])
        built = self.make_result(tasks[1], 12)
        messages: list[str] = []
        with mock.patch.object(pp, "load_valid_shard", side_effect=[reused, None]), mock.patch.object(
            pp, "_run_pending_tasks", return_value=[built]
        ):
            results = pp.process_shards(tasks, "tokenizer.model", 4, True, emit=messages.append)
        self.assertEqual([item.source.index for item in results], [0, 1])
        self.assertTrue(any("reused=1" in line for line in messages))

    def test_process_shards_with_all_reused_results_skips_executor(self) -> None:
        tasks = self.make_tasks(0, 1)
        reused = [self.make_result(task) for task in tasks]
        with mock.patch.object(pp, "load_valid_shard", side_effect=reused), mock.patch.object(
            pp, "_run_pending_tasks", side_effect=AssertionError("executor should not run")
        ):
            results = pp.process_shards(tasks, "tokenizer.model", 4, True)
        self.assertEqual(results, reused)

    def test_process_shards_accepts_resume_tail_indexes(self) -> None:
        tasks = self.make_tasks(4, 5)
        built = [self.make_result(tasks[1]), self.make_result(tasks[0])]
        with mock.patch.object(pp, "load_valid_shard", return_value=None), mock.patch.object(
            pp, "_run_pending_tasks", return_value=built
        ):
            results = pp.process_shards(tasks, "tokenizer.model", 4, True)
        self.assertEqual([item.source.index for item in results], [4, 5])

    def test_process_shards_rejects_duplicate_task_indexes_before_executor(self) -> None:
        tasks = self.make_tasks(4, 4)
        with mock.patch.object(pp, "_run_pending_tasks", side_effect=AssertionError("executor ran")):
            with self.assertRaisesRegex(ValueError, "duplicate source indexes"):
                pp.process_shards(tasks, "tokenizer.model", 4, False, emit=lambda _: None)

    def test_run_pending_tasks_rejects_duplicate_task_indexes_before_executor(self) -> None:
        tasks = self.make_tasks(4, 4)
        with mock.patch.object(pp, "ProcessPoolExecutor", side_effect=AssertionError("executor ran")):
            with self.assertRaisesRegex(ValueError, "duplicate source indexes"):
                pp._run_pending_tasks(tasks, "tokenizer.model", 4, [], 2, 0.0, 1.0, lambda _: None)

    def test_process_shards_rejects_duplicate_and_missing_or_extra_result_indexes(self) -> None:
        tasks = self.make_tasks(4, 5)
        malformed_results = (
            [self.make_result(tasks[0]), self.make_result(tasks[0])],
            [
                self.make_result(tasks[0]),
                pp.ShardResult(
                    pp.FileIdentity(6, "/6.parquet", 10, 20),
                    "/6.bin",
                    "/6.json",
                    2,
                    0,
                    8,
                ),
            ],
        )
        for built in malformed_results:
            with self.subTest(indexes=[item.source.index for item in built]), mock.patch.object(
                pp, "load_valid_shard", return_value=None
            ), mock.patch.object(pp, "_run_pending_tasks", return_value=built):
                with self.assertRaisesRegex(RuntimeError, "shard results do not match"):
                    pp.process_shards(tasks, "tokenizer.model", 4, True, emit=lambda _: None)

    def test_process_shards_empty_task_list_skips_executor(self) -> None:
        with mock.patch.object(pp, "_run_pending_tasks", side_effect=AssertionError("executor ran")):
            self.assertEqual(pp.process_shards([], "tokenizer.model", 4, True, emit=lambda _: None), [])

    def test_direct_api_rejects_invalid_heartbeat_and_worker_counts(self) -> None:
        task = self.make_tasks(0)[0]
        invalid_heartbeats = (0.0, -1.0, float("nan"), float("inf"), float("-inf"))
        for heartbeat in invalid_heartbeats:
            with self.subTest(heartbeat=heartbeat), self.assertRaisesRegex(ValueError, "heartbeat_seconds"):
                pp._run_pending_tasks([task], "tokenizer.model", 1, [], 1, 0.0, heartbeat, lambda _: None)
        for workers in (0, -1):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "num_workers"):
                pp.process_shards([], "tokenizer.model", workers, True, emit=lambda _: None)

    def test_summarize_progress_contains_counts_and_rate(self) -> None:
        result = pp.ShardResult(pp.FileIdentity(0, "/a", 1, 1), "/a.bin", "/a.json", 4, 1, 100)
        message = pp.summarize_progress([result], total=2, started_at=10.0, now=20.0)
        self.assertIn("files=1/2", message)
        self.assertIn("documents=4", message)
        self.assertIn("tokens=100", message)
        self.assertIn("tokens/s=10.0", message)

    def test_run_pending_tasks_emits_heartbeats_on_timeout_and_completion(self) -> None:
        task = self.make_tasks(0)[0]
        result = self.make_result(task)
        future = mock.Mock()
        future.result.return_value = result
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.return_value = future
        messages: list[str] = []
        with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor) as process_pool, mock.patch.object(
            pp, "wait", side_effect=[(set(), {future}), ({future}, set())]
        ), mock.patch.object(pp, "get_context", return_value="spawn-context"):
            results = pp._run_pending_tasks(
                [task], "tokenizer.model", 2, [], 1, 10.0, 0.1, messages.append
            )
        self.assertEqual(results, [result])
        self.assertEqual(len(messages), 2)
        self.assertTrue(all("files=" in message for message in messages))
        executor.submit.assert_called_once_with(pp.tokenize_file, task)
        executor.__enter__.assert_called_once_with()
        process_pool.assert_called_once_with(
            max_workers=2,
            mp_context="spawn-context",
            initializer=pp.initialize_worker,
            initargs=("tokenizer.model",),
        )

    def test_run_pending_tasks_reports_worker_path_and_cancels_remaining_work(self) -> None:
        tasks = self.make_tasks(4, 5)
        failed = mock.Mock()
        failed.result.side_effect = ValueError("bad parquet")
        remaining = mock.Mock()
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [failed, remaining]
        with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor), mock.patch.object(
            pp, "wait", return_value=({failed}, {remaining})
        ), mock.patch.object(pp, "get_context", return_value="spawn-context"):
            with self.assertRaisesRegex(RuntimeError, "failed to preprocess /4.parquet") as raised:
                pp._run_pending_tasks(tasks, "tokenizer.model", 2, [], 2, 10.0, 0.1, lambda _: None)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        remaining.cancel.assert_called_once_with()


class MergeTests(unittest.TestCase):
    def write_result(self, root: Path, index: int, values: list[int]) -> pp.ShardResult:
        shard = root / ".preprocess" / "shards" / f"{index:06d}.bin"
        sidecar = shard.with_suffix(".json")
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(values, dtype="<i4").tofile(shard)
        sidecar.write_text("{}", encoding="utf-8")
        return pp.ShardResult(
            pp.FileIdentity(index, f"/source/{index}.parquet", 1, 1),
            str(shard), str(sidecar), 1, 0, len(values),
        )

    @staticmethod
    def state_payload(manifest: pp.RunManifest, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_sha256": manifest.sha256,
            "next_index": 0,
            "validation_bytes": 0,
            "training_bytes": 0,
            "documents": 0,
            "skipped_documents": 0,
            "tokens": 0,
        }
        payload.update(overrides)
        return payload

    def test_merge_splits_exact_validation_and_deletes_committed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [
                self.write_result(root, 0, [1, 2, 3]),
                self.write_result(root, 1, [4, 5, 6]),
            ]
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", results)
            state = pp.merge_shards(results, root, 4, manifest, resume=False)
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [1, 2, 3, 4])
            self.assertEqual(np.fromfile(root / "train.bin.partial", dtype="<i4").tolist(), [5, 6])
            self.assertEqual(state.next_index, 2)
            self.assertFalse(Path(results[0].shard_path).exists())
            self.assertFalse(Path(results[0].sidecar_path).exists())

    def test_merge_resume_truncates_uncommitted_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.write_result(root, 0, [7])
            second = self.write_result(root, 1, [8, 9])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [first, second])
            np.asarray([7, 99], dtype="<i4").tofile(root / "val.bin.partial")
            pp.atomic_write_json(
                root / ".preprocess" / "merge-state.json",
                self.state_payload(
                    manifest, next_index=1, validation_bytes=4, documents=1, tokens=1
                ),
            )
            pp.merge_shards([first, second], root, 3, manifest, resume=True)
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [7, 8, 9])

    def test_resume_context_does_not_require_deleted_committed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [self.write_result(root, 0, [1]), self.write_result(root, 1, [2])]
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", results)
            pp.atomic_write_json(
                root / ".preprocess" / "merge-state.json",
                self.state_payload(
                    manifest, next_index=1, validation_bytes=4, documents=1, tokens=1
                ),
            )
            Path(results[0].shard_path).unlink()
            Path(results[0].sidecar_path).unlink()
            loaded_manifest, loaded_state = pp.load_resume_context(root / ".preprocess", "run-hash")
            self.assertEqual(loaded_manifest.results, tuple(results))
            self.assertEqual(loaded_state.next_index, 1)

    def test_load_resume_context_requires_both_files_and_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / ".preprocess"
            self.assertIsNone(pp.load_resume_context(state_dir, "run-hash"))
            pp.atomic_write_json(state_dir / "merge-state.json", {})
            with self.assertRaisesRegex(RuntimeError, "both"):
                pp.load_resume_context(state_dir, "run-hash")
            (state_dir / "merge-state.json").unlink()
            manifest = pp.write_run_manifest(state_dir, "run-hash", [])
            with self.assertRaisesRegex(RuntimeError, "both"):
                pp.load_resume_context(state_dir, "run-hash")
            pp.atomic_write_json(state_dir / "merge-state.json", self.state_payload(manifest))
            with self.assertRaisesRegex(RuntimeError, "run hash"):
                pp.load_resume_context(state_dir, "other-run")

    def test_load_resume_context_rejects_invalid_state_types_counters_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            invalid_payloads = (
                self.state_payload(manifest, manifest_sha256="different-manifest"),
                self.state_payload(manifest, next_index=True),
                self.state_payload(manifest, next_index=2),
                self.state_payload(manifest, validation_bytes=2),
                self.state_payload(manifest, validation_bytes=4, tokens=0),
                self.state_payload(manifest, next_index=1, validation_bytes=4, documents=0, tokens=1),
            )
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    pp.atomic_write_json(root / ".preprocess" / "merge-state.json", payload)
                    with self.assertRaises(RuntimeError):
                        pp.load_resume_context(root / ".preprocess", "run-hash")

    def test_write_run_manifest_refuses_a_different_immutable_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".preprocess"
            first = self.write_result(Path(tmp), 0, [1])
            pp.write_run_manifest(state_dir, "run-hash", [first])
            changed = pp.ShardResult(
                first.source, first.shard_path, first.sidecar_path, 1, 0, 2
            )
            with self.assertRaisesRegex(RuntimeError, "differs"):
                pp.write_run_manifest(state_dir, "run-hash", [changed])

    def test_recover_output_file_handles_missing_short_and_long_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.bin"
            pp.recover_output_file(path, 0)
            self.assertTrue(path.exists())
            with self.assertRaisesRegex(RuntimeError, "missing committed"):
                pp.recover_output_file(Path(tmp) / "missing.bin", 4)
            path.write_bytes(b"\x00\x00")
            with self.assertRaisesRegex(RuntimeError, "shorter"):
                pp.recover_output_file(path, 4)
            np.asarray([1, 2], dtype="<i4").tofile(path)
            pp.recover_output_file(path, 4)
            self.assertEqual(path.stat().st_size, 4)

    def test_merge_rejects_wrong_shard_byte_size_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1, 2])
            with Path(result.shard_path).open("r+b") as stream:
                stream.truncate(4)
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            with self.assertRaisesRegex(RuntimeError, "unexpected byte size"):
                pp.merge_shards([result], root, 2, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertTrue(Path(result.sidecar_path).exists())

    def test_merge_keeps_shard_when_journal_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            with mock.patch.object(pp, "atomic_write_json", side_effect=OSError("journal failed")):
                with self.assertRaisesRegex(OSError, "journal failed"):
                    pp.merge_shards([result], root, 1, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertTrue(Path(result.sidecar_path).exists())
            self.assertFalse((root / ".preprocess" / "merge-state.json").exists())
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [1])

    def test_merge_fills_validation_before_training_when_shard_crosses_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [10, 11, 12, 13])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            pp.merge_shards([result], root, 2, manifest, resume=False)
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [10, 11])
            self.assertEqual(np.fromfile(root / "train.bin.partial", dtype="<i4").tolist(), [12, 13])

    def test_merge_commits_an_empty_shard_without_mapping_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            state = pp.merge_shards([result], root, 1, manifest, resume=False)
            self.assertEqual(state.next_index, 1)
            self.assertEqual(state.tokens, 0)
            self.assertFalse(Path(result.shard_path).exists())

    def test_merge_resume_requires_an_existing_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            with self.assertRaisesRegex(RuntimeError, "merge-state.json"):
                pp.merge_shards([result], root, 1, manifest, resume=True)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertFalse((root / "val.bin.partial").exists())

    def test_fresh_merge_refuses_an_existing_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            pp.atomic_write_json(
                root / ".preprocess" / "merge-state.json", self.state_payload(manifest)
            )
            with self.assertRaisesRegex(RuntimeError, "existing merge state"):
                pp.merge_shards([result], root, 1, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertFalse((root / "val.bin.partial").exists())

    def test_fresh_merge_refuses_existing_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            np.asarray([99], dtype="<i4").tofile(root / "val.bin.partial")
            with self.assertRaisesRegex(RuntimeError, "existing partial outputs"):
                pp.merge_shards([result], root, 1, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())

    def test_resume_context_rejects_tampered_generated_shard_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            state_dir = root / ".preprocess"
            pp.write_run_manifest(state_dir, "run-hash", [result])
            manifest_path = state_dir / "run-manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["results"][0]["shard_path"] = str(root / "outside.bin")
            manifest_path.write_text(pp.canonical_json(payload), encoding="utf-8")
            manifest_sha256 = hashlib.sha256(pp.canonical_json(payload).encode("utf-8")).hexdigest()
            pp.atomic_write_json(
                state_dir / "merge-state.json",
                {
                    "manifest_sha256": manifest_sha256,
                    "next_index": 0,
                    "validation_bytes": 0,
                    "training_bytes": 0,
                    "documents": 0,
                    "skipped_documents": 0,
                    "tokens": 0,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "generated shard paths"):
                pp.load_resume_context(state_dir, "run-hash")
            self.assertTrue(Path(result.shard_path).exists())

    def test_write_run_manifest_rejects_tampered_generated_sidecar_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            tampered = pp.ShardResult(
                result.source, result.shard_path, str(root / "outside.json"), 1, 0, 1
            )
            with self.assertRaisesRegex(RuntimeError, "generated shard paths"):
                pp.write_run_manifest(root / ".preprocess", "run-hash", [tampered])

    def test_merge_rejects_result_paths_outside_its_output_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source-output"
            target_root = root / "target-output"
            result = self.write_result(source_root, 0, [1])
            manifest = pp.write_run_manifest(source_root / ".preprocess", "run-hash", [result])
            with self.assertRaisesRegex(RuntimeError, "generated shard paths"):
                pp.merge_shards([result], target_root, 1, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertFalse((target_root / "val.bin.partial").exists())

    def test_atomic_publications_fsync_parent_after_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[tuple[str, Path]] = []
            original_replace = pp.os.replace

            def replace(source: str | Path, target: str | Path) -> None:
                events.append(("replace", Path(target)))
                original_replace(source, target)

            def sync(directory: Path) -> None:
                events.append(("sync", directory))

            with mock.patch.object(pp.os, "replace", replace), mock.patch.object(
                pp, "_fsync_directory", sync
            ):
                pp.atomic_write_json(root / "state.json", {"value": 1})
            self.assertEqual([kind for kind, _ in events[-2:]], ["replace", "sync"])

            events.clear()
            result = self.write_result(root, 0, [1])
            with mock.patch.object(pp.os, "replace", replace), mock.patch.object(
                pp, "_fsync_directory", sync
            ):
                pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            self.assertEqual([kind for kind, _ in events[-2:]], ["replace", "sync"])

    def test_new_partial_output_entry_is_directory_synced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "val.bin.partial"
            events: list[str] = []
            original_touch = Path.touch

            def touch(current: Path, *args: object, **kwargs: object) -> None:
                events.append("touch")
                original_touch(current, *args, **kwargs)

            with mock.patch.object(Path, "touch", touch), mock.patch.object(
                pp, "_fsync_directory", side_effect=lambda _: events.append("sync")
            ):
                pp.recover_output_file(path, 0)
            self.assertEqual(events, ["touch", "sync"])

    def test_directory_fsync_failure_keeps_shard_and_allows_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / ".preprocess"
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(state_dir, "run-hash", [result])

            def fail_state_directory(directory: Path) -> None:
                if directory.resolve() == state_dir.resolve():
                    raise OSError("directory fsync failed")

            with mock.patch.object(pp, "_fsync_directory", side_effect=fail_state_directory):
                with self.assertRaisesRegex(OSError, "directory fsync failed"):
                    pp.merge_shards([result], root, 1, manifest, resume=False)
            self.assertTrue(Path(result.shard_path).exists())
            self.assertTrue((state_dir / "merge-state.json").exists())
            pp.merge_shards([result], root, 1, manifest, resume=True)
            self.assertFalse(Path(result.shard_path).exists())
