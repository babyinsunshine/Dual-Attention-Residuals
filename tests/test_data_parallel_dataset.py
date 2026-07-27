from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from dual_attentions_residuals.data import PackedTokenDataset


class PackedTokenDatasetDataParallelTests(unittest.TestCase):
    """Guard against reverting PackedTokenDataset to map-style DataLoader input."""

    def _token_file(self, directory: Path) -> Path:
        path = directory / "tokens.bin"
        np.arange(96, dtype=np.int32).tofile(path)
        return path

    def _batches_for_rank(self, path: Path, rank: int) -> list[torch.Tensor]:
        dataset = PackedTokenDataset(path, seq_length=3, shuffle_buffer_size=1)
        with patch("dual_attentions_residuals.data._data_parallel_rank_and_world", return_value=(rank, 2)):
            loader = DataLoader(dataset, batch_size=1, num_workers=0)
            iterator = iter(loader)
            return [next(iterator).squeeze(0) for _ in range(4)]

    def test_different_data_parallel_ranks_receive_disjoint_sequence_positions(self) -> None:
        """Changing iteration sharding to sequential map-style indexing must fail this test."""
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._token_file(Path(tmp))
            rank_zero = self._batches_for_rank(tokens, rank=0)
            rank_one = self._batches_for_rank(tokens, rank=1)

        self.assertEqual([batch[0].item() for batch in rank_zero], [0, 6, 12, 18])
        self.assertEqual([batch[0].item() for batch in rank_one], [3, 9, 15, 21])
        self.assertTrue(set(batch[0].item() for batch in rank_zero).isdisjoint(batch[0].item() for batch in rank_one))

    def test_model_parallel_peers_with_same_data_parallel_rank_receive_same_sequence_order(self) -> None:
        """Changing rank selection per model-parallel peer must fail this test."""
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._token_file(Path(tmp))
            first_peer = self._batches_for_rank(tokens, rank=1)
            second_peer = self._batches_for_rank(tokens, rank=1)

        self.assertEqual([batch.tolist() for batch in first_peer], [batch.tolist() for batch in second_peer])

    def test_loader_keeps_direct_sample_format(self) -> None:
        """Changing iterator samples away from seq_length plus one int64 tokens must fail this test."""
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._token_file(Path(tmp))
            sample = self._batches_for_rank(tokens, rank=0)[0]

        self.assertEqual(sample.dtype, torch.int64)
        self.assertEqual(sample.tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
