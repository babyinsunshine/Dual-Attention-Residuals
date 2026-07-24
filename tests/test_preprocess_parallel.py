from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
            original_replace = Path.replace

            def coordinated_replace(source: Path, target: Path) -> Path:
                barrier.wait(timeout=5)
                return original_replace(source, target)

            def write(value: int) -> None:
                try:
                    pp.atomic_write_json(path, {"value": value})
                except Exception as error:  # pragma: no cover - asserted below
                    errors.append(error)

            with mock.patch.object(Path, "replace", coordinated_replace):
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
