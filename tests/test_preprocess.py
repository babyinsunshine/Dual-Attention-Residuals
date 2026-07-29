from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data import preprocess as pp


class DiscoveryTests(unittest.TestCase):
    def test_review_package_defaults(self) -> None:
        args = pp.parse_args([])
        self.assertEqual(
            args.input_glob,
            "data/fineweb_edu_100BT/sample/100BT/*.parquet",
        )
        self.assertEqual(args.output_dir, "data/fineweb_edu_100BT_pretok")
        self.assertTrue(args.add_eos)

    def test_direct_script_help_runs_from_repository_root(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "data/preprocess.py", "--help"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--input-glob", completed.stdout)
        self.assertIn("--num-workers", completed.stdout)
        self.assertIn("--validation-tokens", completed.stdout)
        self.assertIn("--no-add-eos", completed.stdout)
        self.assertIn("data/fineweb_edu_100BT/sample/100BT/*.parquet", completed.stdout)
        self.assertIn("data/fineweb_edu_100BT_pretok", completed.stdout)

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

    def test_parse_args_allows_explicit_eos_opt_out(self) -> None:
        self.assertTrue(pp.parse_args(["--input-glob", "x", "--add-eos"]).add_eos)
        self.assertFalse(pp.parse_args(["--input-glob", "x", "--no-add-eos"]).add_eos)
        with self.assertRaises(SystemExit):
            pp.parse_args(["--input-glob", "x", "--add-eos", "--no-add-eos"])

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

    def test_run_identity_distinguishes_default_eos_from_explicit_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.parquet"
            tokenizer_path = root / "tokenizer.model"
            input_path.write_bytes(b"input")
            tokenizer_path.write_bytes(b"tokenizer")
            with_eos = pp.build_run_identity(
                [input_path], tokenizer_path, pp.EXPECTED_VOCAB_SIZE, "text", False, True
            )
            without_eos = pp.build_run_identity(
                [input_path], tokenizer_path, pp.EXPECTED_VOCAB_SIZE, "text", False, False
            )
            self.assertNotEqual(with_eos.digest(), without_eos.digest())


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

    def test_tokenize_file_rejects_source_changed_before_read_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(root)
            pq.write_table(pa.table({"text": ["changed"]}), task.source.path)
            pp._WORKER_TOKENIZER = FakeTokenizer()
            with self.assertRaisesRegex(RuntimeError, "source identity changed"):
                pp.tokenize_file(task)
            self.assertFalse(Path(task.shard_path).exists())
            self.assertFalse(Path(task.sidecar_path).exists())
            self.assertFalse(Path(task.shard_path + ".partial").exists())

    def test_tokenize_file_rejects_source_replaced_during_read_without_publishing(self) -> None:
        class ReplacingTokenizer(FakeTokenizer):
            def __init__(self, source: Path, replacement: Path) -> None:
                self.source = source
                self.replacement = replacement
                self.replaced = False

            def encode_batch(self, texts, *, add_bos=False, add_eos=False):
                encoded = super().encode_batch(texts, add_bos=add_bos, add_eos=add_eos)
                if not self.replaced:
                    os.replace(self.replacement, self.source)
                    self.replaced = True
                return encoded

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(root)
            replacement = root / "replacement.parquet"
            pq.write_table(pa.table({"text": ["replacement"]}), replacement)
            pp._WORKER_TOKENIZER = ReplacingTokenizer(Path(task.source.path), replacement)
            with self.assertRaisesRegex(RuntimeError, "source identity changed"):
                pp.tokenize_file(task)
            self.assertFalse(Path(task.shard_path).exists())
            self.assertFalse(Path(task.sidecar_path).exists())
            self.assertFalse(Path(task.shard_path + ".partial").exists())


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
        executor.shutdown.assert_called_once_with(wait=True)
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

    def test_run_pending_tasks_cancels_and_reraises_keyboard_interrupt(self) -> None:
        task = self.make_tasks(0)[0]
        interrupted = mock.Mock()
        interrupted.result.side_effect = KeyboardInterrupt("received signal 15")
        remaining = mock.Mock()
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = [interrupted, remaining]
        with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor), mock.patch.object(
            pp, "wait", return_value=({interrupted}, {remaining})
        ), mock.patch.object(pp, "get_context", return_value="spawn-context"):
            with self.assertRaisesRegex(KeyboardInterrupt, "received signal 15"):
                pp._run_pending_tasks([task, self.make_tasks(1)[0]], "tokenizer.model", 2, [], 2, 0.0, 0.1, lambda _: None)
        remaining.cancel.assert_called_once_with()

    def test_run_pending_tasks_does_not_wrap_system_exit(self) -> None:
        task = self.make_tasks(0)[0]
        exited = mock.Mock()
        exited.result.side_effect = SystemExit(3)
        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.return_value = exited
        with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor), mock.patch.object(
            pp, "wait", return_value=({exited}, set())
        ), mock.patch.object(pp, "get_context", return_value="spawn-context"):
            with self.assertRaises(SystemExit) as raised:
                pp._run_pending_tasks([task], "tokenizer.model", 1, [], 1, 0.0, 0.1, lambda _: None)
        self.assertEqual(raised.exception.code, 3)

    def test_run_pending_tasks_aborts_processes_when_submit_fails(self) -> None:
        tasks = self.make_tasks(0, 1)
        submitted = mock.Mock()
        worker = mock.Mock()
        worker.is_alive.return_value = True
        executor = mock.Mock()
        executor.submit.side_effect = [submitted, OSError("submit failed")]
        executor._processes = {1: worker}
        with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor), mock.patch.object(
            pp, "get_context", return_value="spawn-context"
        ):
            with self.assertRaisesRegex(OSError, "submit failed"):
                pp._run_pending_tasks(tasks, "tokenizer.model", 2, [], 2, 0.0, 0.1, lambda _: None)
        submitted.cancel.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        worker.terminate.assert_called_once_with()
        worker.join.assert_called_once_with()

    def test_run_pending_tasks_sigterm_during_wait_cancels_and_terminates_promptly(self) -> None:
        task = self.make_tasks(0)[0]
        future = mock.Mock()
        worker = mock.Mock()
        worker.is_alive.return_value = True
        executor = mock.Mock()
        executor.submit.return_value = future
        executor._processes = {1: worker}

        def deliver_sigterm(*_args: object, **_kwargs: object) -> tuple[set[object], set[object]]:
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("signal handler did not interrupt wait")

        previous = signal.signal(signal.SIGTERM, pp._interrupt_on_sigterm)
        try:
            with mock.patch.object(pp, "ProcessPoolExecutor", return_value=executor), mock.patch.object(
                pp, "wait", side_effect=deliver_sigterm
            ), mock.patch.object(pp, "get_context", return_value="spawn-context"):
                with self.assertRaisesRegex(KeyboardInterrupt, "received signal"):
                    pp._run_pending_tasks([task], "tokenizer.model", 1, [], 1, 0.0, 0.1, lambda _: None)
        finally:
            signal.signal(signal.SIGTERM, previous)
        future.cancel.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        worker.terminate.assert_called_once_with()
        worker.join.assert_called_once_with()

    def test_abort_process_pool_keeps_worker_handles_across_shutdown(self) -> None:
        worker = mock.Mock()
        worker.is_alive.return_value = True
        executor = mock.Mock()
        executor._processes = {1: worker}

        def clear_process_map(*_args: object, **_kwargs: object) -> None:
            executor._processes = None

        executor.shutdown.side_effect = clear_process_map
        pp._abort_process_pool(executor, {})
        worker.terminate.assert_called_once_with()
        worker.join.assert_called_once_with()


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

    def test_load_resume_context_recovers_manifest_only_but_rejects_state_only_and_wrong_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / ".preprocess"
            self.assertIsNone(pp.load_resume_context(state_dir, "run-hash"))
            pp.atomic_write_json(state_dir / "merge-state.json", {})
            with self.assertRaisesRegex(RuntimeError, "both"):
                pp.load_resume_context(state_dir, "run-hash")
            (state_dir / "merge-state.json").unlink()
            manifest = pp.write_run_manifest(state_dir, "run-hash", [])
            loaded_manifest, loaded_state = pp.load_resume_context(state_dir, "run-hash")
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(loaded_state, pp.MergeState(manifest.sha256, 0, 0, 0, 0, 0, 0))
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

    def test_merge_resume_recovers_manifest_only_by_replaying_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.write_result(root, 0, [1])
            manifest = pp.write_run_manifest(root / ".preprocess", "run-hash", [result])
            np.asarray([99], dtype="<i4").tofile(root / "val.bin.partial")
            state = pp.merge_shards([result], root, 1, manifest, resume=True)
            self.assertEqual(state.next_index, 1)
            self.assertEqual(np.fromfile(root / "val.bin.partial", dtype="<i4").tolist(), [1])

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


class FinalizationTests(unittest.TestCase):
    @staticmethod
    def run_identity() -> pp.RunIdentity:
        source = pp.FileIdentity(0, "/source/000.parquet", 1, 1)
        return pp.RunIdentity(1, (source,), "/tokenizer.model", "abc", 131072, "text", False, True)

    @staticmethod
    def manifest(validation_tokens: int = 2, training_tokens: int = 1) -> pp.RunManifest:
        run = FinalizationTests.run_identity()
        source = run.inputs[0]
        results = (pp.ShardResult(
                source, "/output/.preprocess/shards/000000.bin",
                "/output/.preprocess/shards/000000.json", 1, 0,
                validation_tokens + training_tokens,
            ),)
        payload = {"run_sha256": run.digest(), "results": [result.to_json() for result in results]}
        return pp.RunManifest(hashlib.sha256(pp.canonical_json(payload).encode("utf-8")).hexdigest(), results)

    @staticmethod
    def state(validation_tokens: int = 2, training_tokens: int = 1) -> pp.MergeState:
        return pp.MergeState(
            FinalizationTests.manifest(validation_tokens, training_tokens).sha256,
            1, validation_tokens * 4, training_tokens * 4, 1, 0,
            validation_tokens + training_tokens,
        )

    def test_existing_final_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.bin").write_bytes(b"user-data")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                pp.prepare_output(root, overwrite=False, resume=False)
            self.assertEqual((root / "train.bin").read_bytes(), b"user-data")

    def test_existing_preprocess_state_requires_resume_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".preprocess").mkdir()
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                pp.prepare_output(root, overwrite=False, resume=False)

    def test_overwrite_removes_only_owned_names_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            target = outside / "user-data"
            target.write_bytes(b"keep")
            (root / "train.bin").symlink_to(target)
            (root / "unrelated.txt").write_text("keep", encoding="utf-8")
            state = root / ".preprocess"
            state.mkdir()
            (state / "link").symlink_to(outside, target_is_directory=True)
            pp.prepare_output(root, overwrite=True, resume=False)
            self.assertFalse((root / "train.bin").exists())
            self.assertEqual(target.read_bytes(), b"keep")
            self.assertEqual((root / "unrelated.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse(state.exists())

    def test_overwrite_syncs_output_directory_after_named_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.bin").write_bytes(b"generated")
            with mock.patch.object(pp, "_fsync_directory") as sync:
                pp.prepare_output(root, overwrite=True, resume=False)
            self.assertIn(mock.call(root.resolve()), sync.call_args_list)

    def test_overwrite_never_traverses_a_directory_swapped_for_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / ".preprocess"
            victim = state / "victim"
            victim.mkdir(parents=True)
            (victim / "generated").write_text("generated", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            original_scandir = os.scandir
            original_open = os.open
            swapped = False

            def swap() -> None:
                nonlocal swapped
                if not swapped:
                    victim.rename(state / "victim.original")
                    victim.symlink_to(outside, target_is_directory=True)
                    swapped = True

            def race_scandir(path: str | bytes | int | Path):
                if not isinstance(path, int) and Path(path) == victim:
                    swap()
                return original_scandir(path)

            def race_open(path: str | bytes | Path, flags: int, *args: object, **kwargs: object) -> int:
                if path == "victim" and kwargs.get("dir_fd") is not None:
                    swap()
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(pp.os, "scandir", race_scandir), mock.patch.object(pp.os, "open", race_open):
                with self.assertRaises(OSError):
                    pp.prepare_output(root, overwrite=True, resume=False)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_finalize_writes_metadata_last_and_renames_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            metadata = pp.finalize_outputs(
                root, self.run_identity(), self.manifest(), self.state(), batch_size=2,
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
            pp.finalize_outputs(
                root, self.run_identity(), self.manifest(), self.state(), batch_size=2,
                num_workers=1, validation_tokens=2,
            )
            self.assertTrue((root / "metadata.json").is_file())
            self.assertEqual(np.fromfile(root / "val.bin", dtype="<i4").tolist(), [1, 2])
            self.assertEqual(np.fromfile(root / "train.bin", dtype="<i4").tolist(), [3])

    def test_finalize_accepts_same_inode_partial_after_link_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_partial = root / "val.bin.partial"
            np.asarray([1, 2], dtype="<i4").tofile(val_partial)
            os.link(val_partial, root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertTrue(val_partial.exists())
            self.assertEqual(np.fromfile(root / "val.bin", dtype="<i4").tolist(), [1, 2])

    def test_linux_rename_no_replace_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"source")
            destination.write_bytes(b"destination")
            directory_fd = pp._open_output_directory(root)
            try:
                with self.assertRaises(FileExistsError):
                    pp._rename_no_replace(directory_fd, source.name, directory_fd, destination.name)
            finally:
                os.close(directory_fd)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"destination")

    def test_legacy_partial_replacement_after_validation_is_not_unlinked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "val.bin.partial"
            np.asarray([1, 2], dtype="<i4").tofile(partial)
            os.link(partial, root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_open = pp._open_regular_file
            replaced = False

            def open_then_replace(*args: object, **kwargs: object) -> tuple[int, pp.FileIdentityTuple]:
                nonlocal replaced
                descriptor, identity = original_open(*args, **kwargs)
                if not replaced and args[3] == "partial validation output":
                    partial.unlink()
                    np.asarray([90, 91], dtype="<i4").tofile(partial)
                    replaced = True
                return descriptor, identity

            with mock.patch.object(pp, "_open_regular_file", open_then_replace):
                pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual(np.fromfile(partial, dtype="<i4").tolist(), [90, 91])

    def test_finalize_rejects_raced_binary_destination_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_rename = pp._rename_no_replace

            def race(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
                if destination == "val.bin":
                    destination_path = root / destination
                    np.asarray([99, 100], dtype="<i4").tofile(destination_path)
                original_rename(source_fd, source, destination_fd, destination)

            with mock.patch.object(pp, "_rename_no_replace", race):
                with self.assertRaisesRegex(RuntimeError, "destination"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual(np.fromfile(root / "val.bin", dtype="<i4").tolist(), [99, 100])
            self.assertTrue((root / "val.bin.partial").exists())

    def test_finalize_never_replaces_raced_completion_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_rename = pp._rename_no_replace

            def race(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
                destination_path = root / destination
                if destination == "metadata.json":
                    destination_path.write_text('{"race":true}', encoding="utf-8")
                original_rename(source_fd, source, destination_fd, destination)

            with mock.patch.object(pp, "_rename_no_replace", race):
                with self.assertRaisesRegex(FileExistsError, "metadata.json"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"race":true}')
            self.assertEqual(list(root.glob(".metadata.json.*.partial")), [])

    def test_finalize_rejects_partial_swapped_before_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "val.bin.partial"
            np.asarray([1, 2], dtype="<i4").tofile(partial)
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_rename = pp._rename_no_replace

            def swap_then_link(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
                if destination == "val.bin":
                    partial.unlink()
                    np.asarray([90, 91], dtype="<i4").tofile(partial)
                original_rename(source_fd, source, destination_fd, destination)

            with mock.patch.object(pp, "_rename_no_replace", swap_then_link):
                with self.assertRaisesRegex(RuntimeError, "inode"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertFalse((root / "metadata.json").exists())

    def test_finalize_rejects_final_swapped_before_metadata_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin")
            original_create = pp._atomic_create_json

            def swap_then_create(*args: object, **kwargs: object) -> object:
                (root / "val.bin").unlink()
                np.asarray([90, 91], dtype="<i4").tofile(root / "val.bin")
                return original_create(*args, **kwargs)

            with mock.patch.object(pp, "_atomic_create_json", swap_then_create):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"incomplete":true}\n')

    def test_finalize_rolls_back_only_its_metadata_when_final_swaps_after_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin")
            original_create = pp._atomic_create_json

            def create_then_swap(*args: object, **kwargs: object) -> object:
                result = original_create(*args, **kwargs)
                (root / "val.bin").unlink()
                np.asarray([90, 91], dtype="<i4").tofile(root / "val.bin")
                return result

            with mock.patch.object(pp, "_atomic_create_json", create_then_swap):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"incomplete":true}\n')

    def test_finalize_never_unlinks_raced_metadata_during_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin")
            original_create = pp._atomic_create_json

            def create_then_swap_everything(*args: object, **kwargs: object) -> object:
                result = original_create(*args, **kwargs)
                (root / "metadata.json").unlink()
                (root / "metadata.json").write_text('{"racer":true}', encoding="utf-8")
                (root / "val.bin").unlink()
                np.asarray([90, 91], dtype="<i4").tofile(root / "val.bin")
                return result

            with mock.patch.object(pp, "_atomic_create_json", create_then_swap_everything):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"racer":true}')

    def test_finalize_rejects_symlink_substitution_at_validation_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.bin"
            np.asarray([90, 91], dtype="<i4").tofile(outside)
            partial = root / "val.bin.partial"
            np.asarray([1, 2], dtype="<i4").tofile(partial)
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_rename = pp._rename_no_replace

            def symlink_then_link(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
                if destination == "val.bin":
                    partial.unlink()
                    partial.symlink_to(outside)
                original_rename(source_fd, source, destination_fd, destination)

            with mock.patch.object(pp, "_rename_no_replace", symlink_then_link):
                with self.assertRaises(OSError):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertFalse((root / "metadata.json").exists())
            (root / "val.bin").unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            (root / "train.bin.partial").unlink(missing_ok=True)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin")
            original_create = pp._atomic_create_json

            def symlink_then_create(*args: object, **kwargs: object) -> object:
                (root / "val.bin").unlink()
                (root / "val.bin").symlink_to(outside)
                return original_create(*args, **kwargs)

            with mock.patch.object(pp, "_atomic_create_json", symlink_then_create):
                with self.assertRaises(OSError):
                    pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"incomplete":true}\n')

    def test_finalize_keeps_publication_bound_to_open_directory_after_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "output"
            root.mkdir()
            outside = parent / "outside"
            outside.mkdir()
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            original_open = pp._open_output_directory
            original_root = parent / "output.original"
            swapped = False

            def open_then_swap(path: Path) -> int:
                nonlocal swapped
                descriptor = original_open(path)
                if not swapped:
                    root.rename(original_root)
                    root.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return descriptor

            with mock.patch.object(pp, "_open_output_directory", side_effect=open_then_swap):
                metadata = pp.finalize_outputs(
                    root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2
                )
            self.assertEqual(json.loads((original_root / "metadata.json").read_text()), metadata)
            self.assertFalse(any(outside.iterdir()))

    def test_finalize_rejects_insufficient_validation_or_training_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([], dtype="<i4").tofile(root / "train.bin.partial")
            with self.assertRaisesRegex(RuntimeError, "validation"):
                pp.finalize_outputs(
                    root, self.run_identity(), self.manifest(1, 1), self.state(1, 1), 2, 1, 2
                )
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            with self.assertRaisesRegex(RuntimeError, "training"):
                pp.finalize_outputs(
                    root, self.run_identity(), self.manifest(2, 0), self.state(2, 0), 2, 1, 2
                )

    def test_finalize_rejects_partial_final_conflict_and_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin")
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            with self.assertRaisesRegex(RuntimeError, "partial and final"):
                pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            (root / "val.bin.partial").unlink()
            (root / "val.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "byte size"):
                pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)

    def test_finalize_never_replaces_existing_completion_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            (root / "metadata.json").write_text('{"user":"data"}', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "metadata.json"):
                pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual((root / "metadata.json").read_text(encoding="utf-8"), '{"user":"data"}')

    def test_completed_metadata_requires_exact_schema_hashes_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            metadata = pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            self.assertEqual(pp.load_completed_metadata(root, self.run_identity()), metadata)
            for key, value in (("run_sha256", "wrong"), ("train_tokens", 2.0)):
                tampered = dict(metadata)
                tampered[key] = value
                (root / "metadata.json").write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "different output directory"):
                    pp.load_completed_metadata(root, self.run_identity())
            tampered = dict(metadata)
            tampered["unexpected"] = True
            (root / "metadata.json").write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different output directory"):
                pp.load_completed_metadata(root, self.run_identity())
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "train.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "different output directory"):
                pp.load_completed_metadata(root, self.run_identity())

    def test_completed_metadata_closes_descriptors_when_final_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            pp.finalize_outputs(root, self.run_identity(), self.manifest(), self.state(), 2, 1, 2)
            outside = root / "outside.bin"
            np.asarray([90, 91], dtype="<i4").tofile(outside)
            (root / "val.bin").unlink()
            (root / "val.bin").symlink_to(outside)
            before = len(list(Path("/proc/self/fd").iterdir()))
            with self.assertRaisesRegex(RuntimeError, "different output directory"):
                pp.load_completed_metadata(root, self.run_identity())
            after = len(list(Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_finalize_rejects_inconsistent_run_manifest_or_state_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2], dtype="<i4").tofile(root / "val.bin.partial")
            np.asarray([3], dtype="<i4").tofile(root / "train.bin.partial")
            run = self.run_identity()
            source = pp.FileIdentity(0, "/different/000.parquet", 1, 1)
            result = pp.ShardResult(source, "/s.bin", "/s.json", 1, 0, 3)
            payload = {"run_sha256": run.digest(), "results": [result.to_json()]}
            mismatched_manifest = pp.RunManifest(
                hashlib.sha256(pp.canonical_json(payload).encode("utf-8")).hexdigest(), (result,)
            )
            with self.assertRaisesRegex(RuntimeError, "sources"):
                pp.finalize_outputs(
                    root, run, mismatched_manifest,
                    pp.MergeState(mismatched_manifest.sha256, 1, 8, 4, 1, 0, 3), 2, 1, 2,
                )
            inconsistent = pp.MergeState(self.manifest().sha256, 1, 8, 4, 0, 0, 3)
            with self.assertRaisesRegex(RuntimeError, "counters"):
                pp.finalize_outputs(root, run, self.manifest(), inconsistent, 2, 1, 2)
            invalid_result = pp.ShardResult(run.inputs[0], "/s.bin", "/s.json", True, 0, 3)
            invalid_payload = {"run_sha256": run.digest(), "results": [invalid_result.to_json()]}
            invalid_manifest = pp.RunManifest(
                hashlib.sha256(pp.canonical_json(invalid_payload).encode("utf-8")).hexdigest(),
                (invalid_result,),
            )
            invalid_state = pp.MergeState(invalid_manifest.sha256, 1, 8, 4, True, 0, 3)
            with self.assertRaisesRegex(RuntimeError, "negative|invalid"):
                pp.finalize_outputs(root, run, invalid_manifest, invalid_state, 2, 1, 2)
            self.assertTrue((root / "val.bin.partial").exists())
            self.assertTrue((root / "train.bin.partial").exists())


class MainTests(unittest.TestCase):
    def make_completed_resume_case(self, root: Path, *, both_final: bool) -> tuple[list[str], Path]:
        inputs = root / "inputs"
        inputs.mkdir()
        source_path = inputs / "000.parquet"
        pq.write_table(pa.table({"text": ["resume fixture"]}), source_path)
        tokenizer = Path(pp.__file__).resolve().parents[1] / "tokenizer" / "tokenizer.model"
        output = root / "output"
        output.mkdir()
        tokenizer_instance = pp.SentencePieceTokenizer(tokenizer)
        run = pp.build_run_identity([source_path], tokenizer, tokenizer_instance.vocab_size, "text", False, True)
        state_dir = output / ".preprocess"
        task = pp.make_worker_tasks(run, state_dir, 1)[0]
        result = pp.ShardResult(run.inputs[0], task.shard_path, task.sidecar_path, 1, 0, 3)
        manifest = pp.write_run_manifest(state_dir, run.digest(), [result])
        state = pp.MergeState(manifest.sha256, 1, 8, 4, 1, 0, 3)
        pp.atomic_write_json(state_dir / "merge-state.json", pp.asdict(state))
        np.asarray([1, 2], dtype="<i4").tofile(output / "val.bin")
        target = output / ("train.bin" if both_final else "train.bin.partial")
        np.asarray([3], dtype="<i4").tofile(target)
        return [
            "--input-glob", str(inputs / "*.parquet"), "--tokenizer", str(tokenizer),
            "--output-dir", str(output), "--batch-size", "1", "--validation-tokens", "2", "--add-eos", "--resume",
        ], output

    def test_resume_finalizes_one_final_and_one_partial_without_workers_or_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv, output = self.make_completed_resume_case(Path(tmp), both_final=False)
            with mock.patch.object(pp, "process_shards", side_effect=AssertionError("workers ran")), mock.patch.object(
                pp, "merge_shards", side_effect=AssertionError("merge ran")
            ):
                self.assertEqual(pp.main(argv), 0)
            self.assertEqual(np.fromfile(output / "val.bin", dtype="<i4").tolist(), [1, 2])
            self.assertEqual(np.fromfile(output / "train.bin", dtype="<i4").tolist(), [3])
            self.assertTrue((output / "metadata.json").is_file())

    def test_resume_finalizes_two_existing_binaries_without_workers_or_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv, output = self.make_completed_resume_case(Path(tmp), both_final=True)
            with mock.patch.object(pp, "process_shards", side_effect=AssertionError("workers ran")), mock.patch.object(
                pp, "merge_shards", side_effect=AssertionError("merge ran")
            ):
                self.assertEqual(pp.main(argv), 0)
            self.assertEqual(np.fromfile(output / "val.bin", dtype="<i4").tolist(), [1, 2])
            self.assertEqual(np.fromfile(output / "train.bin", dtype="<i4").tolist(), [3])
            self.assertTrue((output / "metadata.json").is_file())

    def test_resume_loads_context_before_processing_tasks(self) -> None:
        args = pp.parse_args(["--input-glob", "unused", "--resume"])
        run = FinalizationTests.run_identity()
        manifest = FinalizationTests.manifest()
        state = pp.MergeState(manifest.sha256, 0, 0, 0, 0, 0, 0)
        events: list[str] = []
        with mock.patch.object(pp, "parse_args", return_value=args), mock.patch.object(
            pp, "prepare_output", return_value=Path("/tmp/output")
        ), mock.patch.object(pp, "discover_inputs", return_value=[]), mock.patch.object(
            pp, "SentencePieceTokenizer", return_value=mock.Mock(vocab_size=131072)
        ), mock.patch.object(pp, "build_run_identity", return_value=run), mock.patch.object(
            pp, "print_disk_status"
        ), mock.patch.object(pp, "load_completed_metadata", return_value=None), mock.patch.object(
            pp, "make_worker_tasks", return_value=[mock.sentinel.task]
        ), mock.patch.object(pp, "load_resume_context", side_effect=lambda *_: (events.append("context") or (manifest, state))), mock.patch.object(
            pp, "process_shards", side_effect=lambda *_: events.append("process") or list(manifest.results)
        ), mock.patch.object(pp, "merge_shards", return_value=state), mock.patch.object(
            pp, "finalize_outputs", return_value={"train_tokens": 1, "validation_tokens": 1}
        ):
            self.assertEqual(pp.main([]), 0)
        self.assertEqual(events, ["context", "process"])

    def test_main_selects_merge_resume_mode_from_loaded_context(self) -> None:
        args = pp.parse_args(["--input-glob", "unused", "--resume"])
        run = FinalizationTests.run_identity()
        manifest = FinalizationTests.manifest()
        incomplete = pp.MergeState(manifest.sha256, 0, 0, 0, 0, 0, 0)
        complete = pp.MergeState(manifest.sha256, 1, 8, 4, 1, 0, 3)
        for context, expected_resume in ((None, False), ((manifest, incomplete), True)):
            with self.subTest(context_loaded=context is not None), mock.patch.object(
                pp, "parse_args", return_value=args
            ), mock.patch.object(
                pp, "prepare_output", return_value=Path("/tmp/output")
            ), mock.patch.object(
                pp, "discover_inputs", return_value=[]
            ), mock.patch.object(
                pp, "SentencePieceTokenizer", return_value=mock.Mock(vocab_size=131072)
            ), mock.patch.object(
                pp, "build_run_identity", return_value=run
            ), mock.patch.object(
                pp, "print_disk_status"
            ), mock.patch.object(
                pp, "load_completed_metadata", return_value=None
            ), mock.patch.object(
                pp, "make_worker_tasks", return_value=[mock.sentinel.task]
            ), mock.patch.object(
                pp, "load_resume_context", return_value=context
            ), mock.patch.object(
                pp, "process_shards", return_value=list(manifest.results)
            ), mock.patch.object(
                pp, "write_run_manifest", return_value=manifest
            ), mock.patch.object(
                pp, "merge_shards", return_value=complete
            ) as merge, mock.patch.object(
                pp, "finalize_outputs", return_value={"train_tokens": 1, "validation_tokens": 1}
            ):
                self.assertEqual(pp.main([]), 0)
                self.assertIs(merge.call_args.args[4], expected_resume)

    def test_main_recovers_manifest_publication_window_and_rebuilds_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            for index, text in enumerate(("alpha beta", "gamma delta")):
                pq.write_table(pa.table({"text": [text]}), inputs / f"{index}.parquet")
            tokenizer = Path(pp.__file__).resolve().parents[1] / "tokenizer" / "tokenizer.model"
            output = pp.prepare_output(root / "output", overwrite=False, resume=True)
            sources = pp.discover_inputs(str(inputs / "*.parquet"), max_files=None)
            run = pp.build_run_identity(sources, tokenizer, pp.EXPECTED_VOCAB_SIZE, "text", False, True)
            tasks = pp.make_worker_tasks(run, output / ".preprocess", batch_size=1)
            previous_tokenizer = pp._WORKER_TOKENIZER
            try:
                pp._WORKER_TOKENIZER = pp.SentencePieceTokenizer(tokenizer)
                results = [pp.tokenize_file(task) for task in tasks]
            finally:
                pp._WORKER_TOKENIZER = previous_tokenizer
            pp.write_run_manifest(output / ".preprocess", run.digest(), results)
            Path(results[0].shard_path).unlink()
            Path(results[0].sidecar_path).unlink()
            np.asarray([999], dtype="<i4").tofile(output / "val.bin.partial")
            np.asarray([999], dtype="<i4").tofile(output / "train.bin.partial")
            exit_code = pp.main([
                "--input-glob", str(inputs / "*.parquet"), "--tokenizer", str(tokenizer),
                "--output-dir", str(output), "--num-workers", "1", "--batch-size", "1",
                "--validation-tokens", "1", "--add-eos", "--resume",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / ".preprocess" / "merge-state.json").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertNotIn(999, np.fromfile(output / "val.bin", dtype="<i4").tolist())
            self.assertNotIn(999, np.fromfile(output / "train.bin", dtype="<i4").tolist())

    def test_sigterm_handler_only_interrupts(self) -> None:
        with self.assertRaisesRegex(KeyboardInterrupt, "received signal 15"):
            pp._interrupt_on_sigterm(15, None)


class EndToEndTests(unittest.TestCase):
    def test_direct_script_resume_on_new_output(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.parquet"
            pq.write_table(pa.table({"text": ["alpha beta", "gamma delta"]}), source)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "data/preprocess.py",
                    "--input-glob", str(source),
                    "--tokenizer", str(repository / "tokenizer" / "tokenizer.model"),
                    "--output-dir", str(output),
                    "--num-workers", "1",
                    "--batch-size", "1",
                    "--validation-tokens", "1",
                    "--add-eos",
                    "--resume",
                ],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "metadata.json").is_file())

    def test_resume_reuses_shard_created_before_manifest_and_starts_fresh_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            for index, text in enumerate(("alpha beta", "gamma delta")):
                pq.write_table(pa.table({"text": [text]}), inputs / f"{index}.parquet")
            tokenizer = Path(pp.__file__).resolve().parents[1] / "tokenizer" / "tokenizer.model"
            output = pp.prepare_output(root / "output", overwrite=False, resume=True)
            sources = pp.discover_inputs(str(inputs / "*.parquet"), max_files=None)
            run = pp.build_run_identity(
                sources, tokenizer, pp.EXPECTED_VOCAB_SIZE, "text", False, True
            )
            tasks = pp.make_worker_tasks(run, output / ".preprocess", batch_size=1)
            previous_tokenizer = pp._WORKER_TOKENIZER
            try:
                pp._WORKER_TOKENIZER = pp.SentencePieceTokenizer(tokenizer)
                reused = pp.tokenize_file(tasks[0])
            finally:
                pp._WORKER_TOKENIZER = previous_tokenizer
            self.assertFalse((output / ".preprocess" / "run-manifest.json").exists())
            self.assertFalse((output / ".preprocess" / "merge-state.json").exists())

            with mock.patch.object(pp, "_run_pending_tasks", wraps=pp._run_pending_tasks) as pending:
                exit_code = pp.main([
                    "--input-glob", str(inputs / "*.parquet"),
                    "--tokenizer", str(tokenizer),
                    "--output-dir", str(output),
                    "--num-workers", "1",
                    "--batch-size", "1",
                    "--validation-tokens", "1",
                    "--add-eos",
                    "--resume",
                ])

            self.assertEqual(exit_code, 0)
            self.assertEqual([task.source.index for task in pending.call_args.args[0]], [1])
            self.assertEqual(pending.call_args.args[3], [reused])
            self.assertTrue((output / "metadata.json").is_file())

    def test_direct_script_runs_single_spawn_worker(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            pq.write_table(pa.table({"text": ["alpha beta", "gamma delta"]}), inputs / "input.parquet")
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "data/preprocess.py",
                    "--input-glob", str(inputs / "*.parquet"),
                    "--tokenizer", str(repository / "tokenizer" / "tokenizer.model"),
                    "--output-dir", str(output),
                    "--num-workers", "1",
                    "--batch-size", "1",
                    "--validation-tokens", "1",
                    "--add-eos",
                ],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "metadata.json").is_file())
            self.assertTrue((output / "train.bin").is_file())
            self.assertTrue((output / "val.bin").is_file())

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
