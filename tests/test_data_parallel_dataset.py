from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from dual_attentions_residuals.data import PackedSampleOrder, PackedTokenDataset


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


def test_resume_matches_uninterrupted_global_sample_order(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(901, dtype=np.int32).tofile(path)
    original = PackedSampleOrder(micro_batch_size=2, num_workers=3, dp_size=2)

    def samples(order, workers, batch_size, consumed, steps):
        rank_iterators = []
        for rank in range(2):
            dataset = PackedTokenDataset(
                path, seq_length=3, start_token=12, shuffle_buffer_size=7,
                micro_batch_size=batch_size, consumed_samples=consumed,
                sample_order=order, data_parallel=(rank, 2),
            )
            rank_iterators.append(iter(DataLoader(dataset, batch_size=batch_size, num_workers=workers)))
        result = []
        for _ in range(steps):
            for iterator in rank_iterators:
                result.extend(next(iterator).tolist())
        return result

    uninterrupted = samples(original, 3, 2, 0, 11)
    direct = PackedTokenDataset(path, seq_length=3, start_token=12, shuffle_buffer_size=7)
    # Original batches cycle through workers 0, 1, 2 on both ranks.
    assert uninterrupted[:12] == [direct[i].tolist() for i in (0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11)]
    # The fifth batch stops in the middle of a three-worker round.
    split = 5 * 2 * 2
    assert samples(original, 3, 2, split, 6) == uninterrupted[split:]
    assert samples(original, 0, 3, split, 4) == uninterrupted[split:]


if __name__ == "__main__":
    unittest.main()
