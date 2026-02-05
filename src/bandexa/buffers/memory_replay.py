from __future__ import annotations

"""
In-memory replay buffer.
The goal is to satisfy the ReplayBufferProtocol defined in buffers/base.py so that
policies (e.g. NeuralLinearTS) can be buffer-backend agnostic.

This buffer:
  - stores experiences in RAM (optionally on GPU if device is cuda)
  - supports uniform sampling for minibatch encoder training
  - supports iter_batches(...) for streaming-style consumption (e.g. posterior rebuild)
    so the policy can use the same logic as disk-backed replay.

Important notes
---------------
- This is a *ring buffer* with a fixed capacity.
- Shapes are inferred on the first add(...) and must remain consistent.
- all() and iter_batches() return experiences in chronological order (oldest -> newest)
  when the buffer has wrapped around.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from bandexa.buffers.base import ReplayBatch, ReplayBufferProtocol

Tensor = torch.Tensor


@dataclass(frozen=True)
class MemoryReplayBatch:
    """
    Concrete batch type for MemoryReplayBuffer.

    Satisfies ReplayBatch Protocol:
      - contexts: Tensor
      - actions:  Tensor
      - rewards:  Tensor
    """
    contexts: Tensor
    actions: Tensor
    rewards: Tensor


class MemoryReplayBuffer(ReplayBufferProtocol):
    """
    A fixed-capacity ring buffer storing (context, action, reward) tensors.

    Required by ReplayBufferProtocol:
      - add(...)
      - sample(...)
      - __len__(...)
      - iter_batches(...)
      - all(...)
    """

    def __init__(self, capacity: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype

        # Allocated lazily on first add() because we don't know shapes yet.
        self._contexts: Tensor | None = None
        self._actions: Tensor | None = None
        self._rewards: Tensor | None = None

        self._ctx_shape: tuple[int, ...] | None = None
        self._act_shape: tuple[int, ...] | None = None

        self._next: int = 0  # next write position in [0, capacity)
        self._size: int = 0  # number of valid entries in [0, capacity]

    def __len__(self) -> int:
        return int(self._size)

    def _ensure_allocated(self, context: Tensor, action: Tensor) -> None:
        if self._contexts is not None:
            # enforce consistent shapes
            if self._ctx_shape is not None and tuple(context.shape) != self._ctx_shape:
                raise ValueError(f"context shape changed: expected {self._ctx_shape}, got {tuple(context.shape)}")
            if self._act_shape is not None and tuple(action.shape) != self._act_shape:
                raise ValueError(f"action shape changed: expected {self._act_shape}, got {tuple(action.shape)}")
            return

        self._ctx_shape = tuple(context.shape)
        self._act_shape = tuple(action.shape)

        self._contexts = torch.empty((self.capacity, *self._ctx_shape), device=self.device, dtype=self.dtype)
        self._actions = torch.empty((self.capacity, *self._act_shape), device=self.device, dtype=self.dtype)
        self._rewards = torch.empty((self.capacity,), device=self.device, dtype=self.dtype)

    def _oldest_physical_index(self) -> int:
        """
        Physical index of the oldest element in the ring.

        If the buffer hasn't wrapped (size < capacity), the oldest is at 0.
        If it has wrapped (size == capacity), the oldest is at self._next.
        """
        if self._size < self.capacity:
            return 0
        return int(self._next)

    def _logical_to_physical(self, logical_idx: Tensor) -> Tensor:
        """
        Map logical indices in [0, size) (oldest->newest) into physical indices in [0, capacity).
        """
        oldest = self._oldest_physical_index()
        # (oldest + logical) % capacity
        return (logical_idx + oldest) % self.capacity

    def add(self, context: Tensor, action: Tensor, reward: float | Tensor) -> None:
        ctx = context.detach().to(device=self.device, dtype=self.dtype)
        act = action.detach().to(device=self.device, dtype=self.dtype)

        if isinstance(reward, torch.Tensor):
            r = reward.detach().to(device=self.device, dtype=self.dtype)
            # accept scalar tensors or (1,) tensors
            if r.numel() != 1:
                raise ValueError(f"reward must be scalar, got shape {tuple(r.shape)}")
            r = r.reshape(())
        else:
            r = torch.tensor(float(reward), device=self.device, dtype=self.dtype)

        self._ensure_allocated(ctx, act)

        assert self._contexts is not None and self._actions is not None and self._rewards is not None

        self._contexts[self._next] = ctx
        self._actions[self._next] = act
        self._rewards[self._next] = r

        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator] = None) -> ReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self._size == 0:
            raise ValueError("buffer is empty")

        assert self._contexts is not None and self._actions is not None and self._rewards is not None

        B = int(batch_size)
        # sample logical indices in [0, size)
        logical = torch.randint(
            low=0,
            high=int(self._size),
            size=(B,),
            generator=generator,
            device=self.device,
        )
        physical = self._logical_to_physical(logical)

        ctx_b = self._contexts.index_select(0, physical)
        act_b = self._actions.index_select(0, physical)
        r_b = self._rewards.index_select(0, physical)

        return MemoryReplayBatch(contexts=ctx_b, actions=act_b, rewards=r_b)

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        **kwargs: object,
    ) -> Iterator[ReplayBatch]:
        """
        Stream experiences in batches.

        The protocol signature in buffers/base.py is:
            iter_batches(batch_size, shuffle=False, generator=None)

        We also accept **kwargs to be tolerant if a policy passes extra flags
        (e.g. include_current=True for some disk buffers).
        """
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self._size == 0:
            return

        assert self._contexts is not None and self._actions is not None and self._rewards is not None

        N = int(self._size)
        logical_order = torch.arange(N, device=self.device)

        if shuffle:
            perm = torch.randperm(N, generator=generator, device=self.device)
            logical_order = logical_order.index_select(0, perm)

        for start in range(0, N, int(batch_size)):
            end = min(start + int(batch_size), N)
            logical = logical_order[start:end]
            physical = self._logical_to_physical(logical)

            yield MemoryReplayBatch(
                contexts=self._contexts.index_select(0, physical),
                actions=self._actions.index_select(0, physical),
                rewards=self._rewards.index_select(0, physical),
            )

    def all(self) -> ReplayBatch:
        """
        Materialize the entire buffer into one batch.

        Returns data in chronological order (oldest -> newest).
        """
        if self._size == 0:
            raise ValueError("buffer is empty")

        assert self._contexts is not None and self._actions is not None and self._rewards is not None

        N = int(self._size)

        # Logical order is oldest->newest; map to physical and gather.
        logical = torch.arange(N, device=self.device)
        physical = self._logical_to_physical(logical)

        return MemoryReplayBatch(
            contexts=self._contexts.index_select(0, physical),
            actions=self._actions.index_select(0, physical),
            rewards=self._rewards.index_select(0, physical),
        )


__all__ = ["MemoryReplayBuffer", "MemoryReplayBatch"]
