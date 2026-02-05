from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

Tensor = torch.Tensor


@runtime_checkable
class Encoder(Protocol):
    """
    Minimal encoder interface for NeuralLinearTS.

    The policy only needs a feature vector z = phi(context, action) in R^d.

    Implementations are typically torch.nn.Module's, but this is intentionally
    kept architecture-agnostic (concat, two-tower, transformers, etc.)
    """

    def encode(self, context: Tensor, action: Tensor, **kwargs: object) -> Tensor:
        """
        Encode a single (context, action) pair.

        Args:
            context: Tensor representing the context. Shape is user-defined.
            action: Tensor representing the action. Shape is user-defined.

        Returns:
            z: feature vector of shape (d,)
        """
        ...


@runtime_checkable
class BatchEncoder(Encoder, Protocol):
    """
    Optional capability: vectorized encoding across many candidate actions.

    If provided, the policy can use this for speed when scoring many actions.
    """

    def encode_batch(self, context: Tensor, actions: Tensor, **kwargs: object) -> Tensor:
        """
        Encode a batch of candidate actions for the same context.

        Args:
            context: Tensor for the context. Shape is user-defined.
            actions: Tensor for candidate actions, typically shape (n_actions, ...)

        Returns:
            Z: feature matrix of shape (n_actions, d)
        """
        ...


def has_encode_batch(encoder: object) -> bool:
    """Return True if encoder supports the optional BatchEncoder capability."""
    return isinstance(encoder, BatchEncoder)
