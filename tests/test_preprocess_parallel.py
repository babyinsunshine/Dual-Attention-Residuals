from __future__ import annotations

import json
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
