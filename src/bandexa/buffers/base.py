from __future__ import annotations

"""
Buffer abstraction + factory.
In real settings you often want:
  - disk-backed replay (for large contexts like images / long sequences)
  - hybrid replay (recent in RAM + older on disk)
  - streaming / online logs (consume from Kafka/S3/etc.)
  - sharded replay (data-parallel training)

To enable that evolution without repeatedly editing NeuralLinearTS, we define:

  1) A *Protocol* describing the minimal buffer interface the policy expects
  2) A small config schema
  3) A factory function that builds a buffer from the config

Important design notes
----------------------
- We intentionally prefer Protocols over base-class inheritance:
  different buffer implementations (memory vs disk) have very different internal concerns
  and often do not share meaningful state or code.

- We standardize on a minimal "batch" shape:
    batch.contexts: Tensor
    batch.actions: Tensor
    batch.rewards: Tensor
  The concrete batch type can differ per buffer as long as it provides these attributes.

- We include an iterator-style API (iter_batches) because disk buffers should *not*
  be forced to materialize everything into memory (i.e., avoid `.all()` when huge).
  Memory buffers can implement iter_batches easily; disk buffers can stream from storage.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional, Protocol, TypeGuard, Union, runtime_checkable

import torch

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Batch protocol (structural typing)
# -----------------------------------------------------------------------------

@runtime_checkable
class ReplayBatch(Protocol):
    """
    Minimal structural contract for a sampled batch.

    Any object with these attributes is accepted (dataclass, NamedTuple, simple class, etc.).
    Shapes are implementation-defined, but typical expectations are:
      - contexts: (B, *context_shape)
      - actions:  (B, *action_shape)
      - rewards:  (B,)  (float)  OR  (B,1) depending on posteror/loss
    """
    contexts: Tensor
    actions: Tensor
    rewards: Tensor


# -----------------------------------------------------------------------------
# Buffer protocol (structural typing)
# -----------------------------------------------------------------------------

@runtime_checkable
class ReplayBufferProtocol(Protocol):
    """
    Minimal interface used by policies (e.g. NeuralLinearTS).

    Required methods:
      - add(...)            : log one experience
      - sample(batch_size)  : sample a random minibatch for encoder training
      - __len__             : number of stored experiences

    Strongly recommended:
      - iter_batches(...)   : stream experiences in chunks (useful for posterior rebuild)

    Optional:
      - all()               : materialize everything (only safe for small-ish buffers)
    """

    def add(self, context: Tensor, action: Tensor, reward: float | Tensor) -> None:
        """Add one (context, action, reward) tuple to the replay."""

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator] = None) -> ReplayBatch:
        """Sample a minibatch uniformly (or implementation-defined) from stored experiences."""

    def __len__(self) -> int:
        """Number of stored experiences."""

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[ReplayBatch]:
        """
        Stream experiences in batches.

        This is the key method for disk-backed replay buffers to support "rebuild posterior"
        without forcing an O(N) in-memory materialization.
        """

    def all(self) -> ReplayBatch:
        """
        Materialize the entire buffer into one batch.

        Not recommended for large buffers; provided for convenience.
        Disk buffers may choose not to implement this.
        """


# -----------------------------------------------------------------------------
# Config schema
# -----------------------------------------------------------------------------

BufferKind = Literal["memory", "disk"]


@dataclass(frozen=True)
class MemoryReplayConfig:
    """
    In-memory replay buffer config.

    capacity:
      Maximum number of experiences retained. When full, the oldest samples are evicted
      (typical ring-buffer behavior), depending on the concrete implementation.
    """
    kind: Literal["memory"] = "memory"
    capacity: int = 50_000


@dataclass(frozen=True)
class DiskReplayConfig:
    """
    Disk-backed replay buffer config.

    root_dir:
      Directory where replay shards/files live.

    capacity:
      Logical capacity in number of experiences retained. Concrete implementations may:
        - enforce it strictly (delete old shards / overwrite)
        - or treat it as a soft cap

    Note:
      Disk buffers commonly also have parameters like shard_size, compression, dtype policy,
      prefetch count, etc. Keep the base config minimal; add more fields as needed without
      changing the policy.
    """
    kind: Literal["disk"] = "disk"
    root_dir: str | PathLike[str] = "bandexa_replay"
    capacity: int = 500_000


BufferConfig = Union[MemoryReplayConfig, DiskReplayConfig]


def _is_memory_cfg(cfg: BufferConfig) -> TypeGuard[MemoryReplayConfig]:
    return isinstance(cfg, MemoryReplayConfig) or getattr(cfg, "kind", None) == "memory"


def _is_disk_cfg(cfg: BufferConfig) -> TypeGuard[DiskReplayConfig]:
    return isinstance(cfg, DiskReplayConfig) or getattr(cfg, "kind", None) == "disk"


def parse_buffer_config(obj: Mapping[str, Any]) -> BufferConfig:
    """
    Helper to construct a BufferConfig from a dict-like payload.

    This is useful when configs come from JSON/YAML, CLI args, or service configs.

    Examples:
      parse_buffer_config({"kind": "memory", "capacity": 10000})
      parse_buffer_config({"kind": "disk", "root_dir": "/tmp/replay", "capacity": 200000})
    """
    kind = obj.get("kind", "memory")
    if kind == "memory":
        return MemoryReplayConfig(capacity=int(obj.get("capacity", 50_000)))
    if kind == "disk":
        return DiskReplayConfig(
            root_dir=obj.get("root_dir", "bandexa_replay"),
            capacity=int(obj.get("capacity", 500_000)),
        )
    raise ValueError(f"Unknown buffer kind: {kind!r}")


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_replay_buffer(
    cfg: BufferConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ReplayBufferProtocol:
    """
    Build a replay buffer from config.

    We import concrete implementations lazily so:
      - importing bandexa.buffers.base does not pull in optional heavy deps,
      - and you can ship disk buffer as optional without breaking memory-only usage.

    Expected concrete modules/classes
    --------------------------------
    - Memory buffer:
        bandexa.buffers.memory_replay.MemoryReplayBuffer

    - Disk buffer:
        bandexa.buffers.disk_replay.DiskReplayBuffer   (recommended name)
      If your disk buffer uses a different class name, update the import below accordingly.

    Constructor flexibility
    -----------------------
    Disk buffer APIs vary. We try a couple of common constructor signatures to reduce
    coupling between the factory and the implementation.
    """
    if _is_memory_cfg(cfg):
        from bandexa.buffers.memory_replay import MemoryReplayBuffer  # existing in-memory buffer

        if cfg.capacity <= 0:
            raise ValueError(f"MemoryReplayConfig.capacity must be > 0, got {cfg.capacity}")

        return MemoryReplayBuffer(int(cfg.capacity), device=device, dtype=dtype)

    if _is_disk_cfg(cfg):
        if int(cfg.capacity) <= 0:
            raise ValueError(f"DiskReplayConfig.capacity must be > 0, got {cfg.capacity}")

        root = Path(cfg.root_dir)

        # Try common class name/signatures. Adjust here if your disk_replay.py differs.
        from bandexa.buffers.disk_replay import DiskReplayBuffer  # you created this module

        # We try multiple signatures so disk_replay.py can evolve without constant churn here.
        # 1) (root_dir, capacity, device, dtype)
        try:
            return DiskReplayBuffer(root_dir=root, capacity=int(cfg.capacity), device=device, dtype=dtype)  # type: ignore[call-arg]
        except TypeError:
            pass

        # 2) (root_dir, capacity) where device/dtype are handled during sampling
        try:
            return DiskReplayBuffer(root_dir=root, capacity=int(cfg.capacity))  # type: ignore[call-arg]
        except TypeError:
            pass

        # 3) (root_dir, max_size, device, dtype) alternate naming
        try:
            return DiskReplayBuffer(root_dir=root, max_size=int(cfg.capacity), device=device, dtype=dtype)  # type: ignore[call-arg]
        except TypeError as e:
            raise TypeError(
                "DiskReplayBuffer constructor signature did not match any supported patterns. "
                "Please adapt make_replay_buffer() to your DiskReplayBuffer API."
            ) from e

    # This should be unreachable if cfg is a proper BufferConfig
    raise TypeError(f"Unsupported buffer config type: {type(cfg)}")


__all__ = [
    "ReplayBatch",
    "ReplayBufferProtocol",
    "BufferKind",
    "MemoryReplayConfig",
    "DiskReplayConfig",
    "BufferConfig",
    "parse_buffer_config",
    "make_replay_buffer",
]
