# src/bandexa/buffers/disk_dataset.py
from __future__ import annotations

from typing import Iterator, Optional
import torch
from torch.utils.data import IterableDataset

from bandexa.buffers.disk_replay import DiskReplayBuffer, ReplayBatch


class DiskReplayDataset(IterableDataset):
    """
    Infinite iterable dataset that yields random minibatches from DiskReplayBuffer.
    Good for SGD training loops that want `for batch in loader: ...`.
    """

    def __init__(self, buffer: DiskReplayBuffer, *, batch_size: int, generator: Optional[torch.Generator] = None):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.generator = generator

    def __iter__(self) -> Iterator[ReplayBatch]:
        while True:
            yield self.buffer.sample(self.batch_size, generator=self.generator)
