from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union
import os
import json
import bisect
from collections import OrderedDict

import torch

Tensor = torch.Tensor
PathLike = Union[str, os.PathLike[str]]


@dataclass
class ReplayBatch:
    """A minibatch of experiences."""
    contexts: Tensor
    actions: Tensor
    rewards: Tensor


class DiskReplayBuffer:
    """
    Disk-backed replay buffer with sharded storage.

    - Writes experiences into fixed-size shard tensors (stored as .pt files).
    - Sampling does not require loading the whole dataset into RAM.
    - Keeps the "current" (not-yet-flushed) shard in memory and includes it in sampling/training.

    Layout:
      root/
        index.json
        shards/
          shard_000000.pt
          shard_000001.pt
          ...

    Notes:
      - Assumes context/action shapes are consistent across inserts.
      - Uses atomic writes (write tmp then os.replace).
      - No external deps.
    """

    def __init__(
        self,
        root_dir: PathLike,
        *,
        shard_size: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        cache_shards: int = 2,
    ) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if cache_shards <= 0:
            raise ValueError(f"cache_shards must be positive, got {cache_shards}")

        self.root = Path(root_dir)
        self.shards_dir = self.root / "shards"
        self.index_path = self.root / "index.json"

        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = int(shard_size)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # On-disk index
        self._shard_files: list[str] = []
        self._shard_counts: list[int] = []
        self._cum_counts: list[int] = []  # prefix sums (end indices)

        # Current (in-progress) shard (kept in memory, CPU)
        self._ctx_shape: tuple[int, ...] | None = None
        self._act_shape: tuple[int, ...] | None = None
        self._cur_ctx: Tensor | None = None
        self._cur_act: Tensor | None = None
        self._cur_rew: Tensor | None = None
        self._cur_count: int = 0
        self._next_shard_id: int = 0

        # Simple LRU cache for loaded shards
        self._cache: "OrderedDict[int, tuple[Tensor, Tensor, Tensor, int]]" = OrderedDict()
        self._cache_limit = int(cache_shards)

        self._load_index()

    def __len__(self) -> int:
        return int(sum(self._shard_counts) + self._cur_count)

    # -------------------------
    # Public API
    # -------------------------

    def add(self, context: Tensor, action: Tensor, reward: float | Tensor) -> None:
        """
        Add one experience.
        Stores on CPU in float dtype (contexts/actions cast to self.dtype; reward cast to float32).
        """
        ctx = self._to_cpu_tensor(context, dtype=self.dtype)
        act = self._to_cpu_tensor(action, dtype=self.dtype)

        if isinstance(reward, torch.Tensor):
            rew = reward.detach().to(device=torch.device("cpu"), dtype=torch.float32).reshape(())
        else:
            rew = torch.tensor(float(reward), device=torch.device("cpu"), dtype=torch.float32)

        self._ensure_current_allocated(ctx, act)

        assert self._cur_ctx is not None and self._cur_act is not None and self._cur_rew is not None

        if self._cur_count >= self.shard_size:
            self.flush()

        # Insert
        self._cur_ctx[self._cur_count].copy_(ctx)
        self._cur_act[self._cur_count].copy_(act)
        self._cur_rew[self._cur_count].copy_(rew)
        self._cur_count += 1

        # Auto-flush if shard filled
        if self._cur_count >= self.shard_size:
            self.flush()

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator] = None) -> ReplayBatch:
        """
        Uniform random sampling over ALL items (disk shards + current shard).
        Returns tensors on (self.device, self.dtype).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = len(self)
        if n == 0:
            raise ValueError("buffer is empty")

        # Choose global indices
        if generator is None:
            idx = torch.randint(low=0, high=n, size=(batch_size,), device="cpu")
        else:
            idx = torch.randint(low=0, high=n, size=(batch_size,), device="cpu", generator=generator)

        ctxs, acts, rews = self._gather_indices(idx.tolist())
        return ReplayBatch(
            contexts=ctxs.to(device=self.device, dtype=self.dtype),
            actions=acts.to(device=self.device, dtype=self.dtype),
            rewards=rews.to(device=self.device, dtype=torch.float32),
        )

    def iter_batches(
        self,
        batch_size: int,
        *,
        include_current: bool = True,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[ReplayBatch]:
        """
        Iterate over the entire dataset as minibatches.

        - shuffle=False: streams sequentially shard-by-shard (disk-efficient).
        - shuffle=True: yields endless random batches using sample() (good for SGD).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if shuffle:
            while True:
                yield self.sample(batch_size, generator=generator)
        else:
            # Disk shards
            for shard_idx in range(len(self._shard_files)):
                ctx, act, rew, count = self._load_shard(shard_idx)
                start = 0
                while start < count:
                    end = min(start + batch_size, count)
                    yield ReplayBatch(
                        contexts=ctx[start:end].to(self.device, self.dtype),
                        actions=act[start:end].to(self.device, self.dtype),
                        rewards=rew[start:end].to(self.device, torch.float32),
                    )
                    start = end

            # Current in-memory shard
            if include_current and self._cur_count > 0:
                assert self._cur_ctx is not None and self._cur_act is not None and self._cur_rew is not None
                start = 0
                while start < self._cur_count:
                    end = min(start + batch_size, self._cur_count)
                    yield ReplayBatch(
                        contexts=self._cur_ctx[start:end].to(self.device, self.dtype),
                        actions=self._cur_act[start:end].to(self.device, self.dtype),
                        rewards=self._cur_rew[start:end].to(self.device, torch.float32),
                    )
                    start = end

    def flush(self) -> None:
        """Flush current shard to disk if it has any items."""
        if self._cur_count == 0:
            return
        assert self._cur_ctx is not None and self._cur_act is not None and self._cur_rew is not None

        shard_name = f"shard_{self._next_shard_id:06d}.pt"
        shard_path = self.shards_dir / shard_name
        tmp_path = self.shards_dir / f".{shard_name}.tmp"

        payload = {
            "contexts": self._cur_ctx[: self._cur_count].contiguous(),
            "actions": self._cur_act[: self._cur_count].contiguous(),
            "rewards": self._cur_rew[: self._cur_count].contiguous(),
            "count": int(self._cur_count),
        }

        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(shard_path))  # atomic

        # Update index (atomic)
        self._shard_files.append(shard_name)
        self._shard_counts.append(int(self._cur_count))
        self._rebuild_cumsums()
        self._next_shard_id += 1
        self._write_index_atomic()

        # Reset current shard (keep shapes)
        self._cur_count = 0

        # Clear cache (optional; keeps behavior simple)
        self._cache.clear()

    # -------------------------
    # Internals
    # -------------------------

    def _to_cpu_tensor(self, x: Tensor, *, dtype: torch.dtype) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("context/action must be torch.Tensor")
        return x.detach().to(device=torch.device("cpu"), dtype=dtype)

    def _ensure_current_allocated(self, ctx: Tensor, act: Tensor) -> None:
        if self._ctx_shape is None:
            self._ctx_shape = tuple(ctx.shape)
            self._act_shape = tuple(act.shape)

            # allocate full shard tensors on CPU
            self._cur_ctx = torch.empty((self.shard_size, *self._ctx_shape), device="cpu", dtype=self.dtype)
            self._cur_act = torch.empty((self.shard_size, *self._act_shape), device="cpu", dtype=self.dtype)
            self._cur_rew = torch.empty((self.shard_size,), device="cpu", dtype=torch.float32)
            return

        if tuple(ctx.shape) != self._ctx_shape:
            raise ValueError(f"context shape mismatch: expected {self._ctx_shape}, got {tuple(ctx.shape)}")
        if tuple(act.shape) != self._act_shape:
            raise ValueError(f"action shape mismatch: expected {self._act_shape}, got {tuple(act.shape)}")

        if self._cur_ctx is None or self._cur_act is None or self._cur_rew is None:
            # shapes known, but tensors missing (shouldn't happen)
            self._cur_ctx = torch.empty((self.shard_size, *self._ctx_shape), device="cpu", dtype=self.dtype)
            self._cur_act = torch.empty((self.shard_size, *self._act_shape), device="cpu", dtype=self.dtype)
            self._cur_rew = torch.empty((self.shard_size,), device="cpu", dtype=torch.float32)

    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._next_shard_id = 0
            return

        data = json.loads(self.index_path.read_text())
        self._shard_files = list(data.get("shards", []))
        self._shard_counts = list(data.get("counts", []))
        self._next_shard_id = int(data.get("next_shard_id", len(self._shard_files)))
        self._rebuild_cumsums()

    def _write_index_atomic(self) -> None:
        tmp = self.root / ".index.json.tmp"
        data = {
            "version": 1,
            "shards": self._shard_files,
            "counts": self._shard_counts,
            "next_shard_id": self._next_shard_id,
        }
        tmp.write_text(json.dumps(data))
        os.replace(str(tmp), str(self.index_path))  # atomic

    def _rebuild_cumsums(self) -> None:
        self._cum_counts = []
        s = 0
        for c in self._shard_counts:
            s += int(c)
            self._cum_counts.append(s)

    def _find_shard(self, global_idx: int) -> tuple[str, int]:
        """
        Map global index in [0, len(disk_shards)) to (shard_file, offset).
        Excludes current in-memory shard.
        """
        shard_idx = bisect.bisect_right(self._cum_counts, global_idx)
        prev_end = 0 if shard_idx == 0 else self._cum_counts[shard_idx - 1]
        offset = global_idx - prev_end
        return self._shard_files[shard_idx], int(offset)

    def _load_shard(self, shard_idx: int) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        Load shard tensors from disk (CPU) with small LRU cache.
        Returns (contexts, actions, rewards, count).
        """
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]

        shard_name = self._shard_files[shard_idx]
        path = self.shards_dir / shard_name
        payload = torch.load(str(path), map_location="cpu")

        ctx: Tensor = payload["contexts"]
        act: Tensor = payload["actions"]
        rew: Tensor = payload["rewards"]
        count = int(payload.get("count", ctx.shape[0]))

        self._cache[shard_idx] = (ctx, act, rew, count)
        self._cache.move_to_end(shard_idx)

        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

        return ctx, act, rew, count

    def _gather_indices(self, indices: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Gather rows for global indices over (disk_shards + current_shard).
        Returns CPU tensors.
        """
        disk_n = int(sum(self._shard_counts))
        ctx_rows: list[Tensor] = []
        act_rows: list[Tensor] = []
        rew_rows: list[Tensor] = []

        for gi in indices:
            if gi < disk_n:
                # disk-backed
                shard_idx = bisect.bisect_right(self._cum_counts, gi)
                prev_end = 0 if shard_idx == 0 else self._cum_counts[shard_idx - 1]
                off = gi - prev_end
                ctx, act, rew, count = self._load_shard(shard_idx)
                if off >= count:
                    raise RuntimeError("index mapping bug: offset beyond shard count")
                ctx_rows.append(ctx[off])
                act_rows.append(act[off])
                rew_rows.append(rew[off])
            else:
                # current in-memory shard
                off = gi - disk_n
                if off >= self._cur_count:
                    raise RuntimeError("index mapping bug: offset beyond current count")
                assert self._cur_ctx is not None and self._cur_act is not None and self._cur_rew is not None
                ctx_rows.append(self._cur_ctx[off])
                act_rows.append(self._cur_act[off])
                rew_rows.append(self._cur_rew[off])

        ctxs = torch.stack(ctx_rows, dim=0)
        acts = torch.stack(act_rows, dim=0)
        rews = torch.stack(rew_rows, dim=0).reshape(-1)
        return ctxs, acts, rews
