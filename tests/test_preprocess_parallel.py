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
