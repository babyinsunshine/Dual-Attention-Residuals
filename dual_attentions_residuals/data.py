from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import IterableDataset


class PackedTokenDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        seq_length: int,
        seed: int = 42,
        start_token: int = 0,
        shuffle_buffer_size: int = 10000,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.seq_length = int(seq_length)
        self.chunk_size = self.seq_length + 1
        self.seed = int(seed)
        self.start_sample = int(start_token) // self.seq_length
        self.shuffle_window = max(1, int(shuffle_buffer_size))
        self.tokens = np.memmap(self.path, dtype=np.int32, mode="r")
        if self.tokens.shape[0] <= self.chunk_size:
            raise ValueError(f"{self.path} does not contain enough tokens for seq_length={seq_length}")
        self.num_sequences = max(1, (self.tokens.shape[0] - 1) // self.seq_length)
        self._effective_window = min(self.shuffle_window, self.num_sequences)
        self._windows_per_epoch = math.ceil(self.num_sequences / self._effective_window)
        self._window_cache: dict[tuple[int, int], tuple[int, int]] = {}

    def __len__(self) -> int:
        # Megatron's official DataLoader expects map-style datasets to be long
        # enough for the requested training schedule. __getitem__ wraps through
        # the token stream, so expose a virtual length to avoid StopIteration in
        # short pretokenized smoke tests and speed sweeps.
        return max(self.num_sequences, 1_000_000_000)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence_index = self._sequence_index(self.start_sample + int(index))
        start = sequence_index * self.seq_length
        sample = np.asarray(self.tokens[start : start + self.chunk_size], dtype=np.int64)
        return torch.from_numpy(sample)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        rank, world = _data_parallel_rank_and_world()
        num_shards = world * num_workers
        shard_id = rank * num_workers + worker_id
        sample_index = self.start_sample + shard_id
        while True:
            sequence_index = self._sequence_index(sample_index)
            start = sequence_index * self.seq_length
            sample = np.asarray(self.tokens[start : start + self.chunk_size], dtype=np.int64)
            yield torch.from_numpy(sample)
            sample_index += num_shards

    def _sequence_index(self, sample_index: int) -> int:
        if self.shuffle_window <= 1:
            return sample_index % self.num_sequences

        epoch_index, epoch_offset = divmod(sample_index, self.num_sequences)
        window_index, window_offset = divmod(epoch_offset, self._effective_window)
        window_start = window_index * self._effective_window
        window_length = min(self._effective_window, self.num_sequences - window_start)
        permutation_index = (epoch_index * self._windows_per_epoch) + window_index
        cache_key = (window_length, permutation_index)
        stride, shift = self._window_cache.get(cache_key, (0, 0))
        if stride == 0:
            stride, shift = _window_permutation(window_length, self.seed, permutation_index)
            self._window_cache[cache_key] = (stride, shift)
        shuffled_offset = (window_offset * stride + shift) % window_length
        return window_start + shuffled_offset


def batch_to_tokens_and_labels(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def _data_parallel_rank_and_world() -> tuple[int, int]:
    if not torch.distributed.is_initialized():
        return 0, 1
    try:
        from megatron.core import parallel_state

        return int(parallel_state.get_data_parallel_rank()), int(parallel_state.get_data_parallel_world_size())
    except Exception:
        return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _window_permutation(window_size: int, seed: int, window_index: int) -> tuple[int, int]:
    if window_size <= 1:
        return 1, 0

    mixed = _splitmix64((seed << 1) ^ window_index)
    shift = mixed % window_size
    stride = 1 + ((mixed >> 32) % max(1, window_size - 1))
    while math.gcd(stride, window_size) != 1:
        stride += 1
        if stride >= window_size:
            stride = 1
    return stride, shift


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask
