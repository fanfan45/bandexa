# This python module is a stub
# TODO implement the Bayesian Logistic Regression Posterior

# src/bandexa/posterior/bayes_logistic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BLOGRConfig:
    """Configuration stub for Bayesian logistic regression posterior."""
    dim: int
    prior_var: float = 1.0
    jitter: float = 1e-6
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


class BayesianLogisticRegression:
    """
    Stub for future Bayesian logistic regression posterior.

    Intended contract (for use in NeuralLinearTS):
      - sample_weights(n_samples=1, generator=None) -> Tensor (d,) or (n_samples, d)
      - update(z, r) updates posterior with one or many observations
      - posterior_mean() -> Tensor (d,)
      - state_dict() / load_state_dict() for inference checkpointing
    """

    def __init__(
        self,
        dim: int,
        *,
        prior_var: float = 1.0,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = BLOGRConfig(
            dim=int(dim),
            prior_var=float(prior_var),
            jitter=float(jitter),
            dtype=dtype,
            device=device,
        )
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        raise NotImplementedError(
            "BayesianLogisticRegression is a stub. "
            "Implement Laplace/VI/EP posterior + sampling before using."
        )

    @property
    def dim(self) -> int:
        return self.config.dim

    def posterior_mean(self) -> torch.Tensor:
        raise NotImplementedError

    def update(self, z: torch.Tensor, r: torch.Tensor | float) -> None:
        raise NotImplementedError

    def sample_weights(
        self,
        *,
        n_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def mean_score(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def sample_score(
        self,
        z: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, object]) -> None:
        raise NotImplementedError
