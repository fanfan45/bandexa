# src/bandexa/posterior/bayes_linear.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BLRConfig:
    """Configuration for Bayesian linear regression posterior."""
    dim: int
    prior_var: float = 1.0          # scalar prior variance (isotropic)
    obs_noise_var: float = 1.0      # observation noise variance
    jitter: float = 1e-6            # added to precision diagonal for Cholesky stability
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


class BayesianLinearRegression:
    """
    Conjugate Bayesian linear regression with Gaussian prior and Gaussian noise.

    Model:
        w ~ N(mu0, Sigma0)  where Sigma0 = prior_var * I
        r | z, w ~ N(z^T w, obs_noise_var)

    We maintain the posterior in *precision* form:
        Lambda = Sigma^{-1}
        b = Lambda * mu

    Posterior update (batch):
        Lambda <- Lambda + (Z^T Z) / obs_noise_var
        b      <- b      + (Z^T r) / obs_noise_var
    """

    def __init__(
        self,
        dim: int,
        *,
        prior_var: float = 1.0,
        obs_noise_var: float = 1.0,
        prior_mean: Optional[torch.Tensor] = None,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if prior_var <= 0:
            raise ValueError(f"prior_var must be > 0, got {prior_var}")
        if obs_noise_var <= 0:
            raise ValueError(f"obs_noise_var must be > 0, got {obs_noise_var}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")

        self.config = BLRConfig(
            dim=dim,
            prior_var=float(prior_var),
            obs_noise_var=float(obs_noise_var),
            jitter=float(jitter),
            dtype=dtype,
            device=device,
        )
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        mu0 = (
            prior_mean.to(device=self.device, dtype=self.dtype)
            if prior_mean is not None
            else torch.zeros(dim, device=self.device, dtype=self.dtype)
        )
        if mu0.shape != (dim,):
            raise ValueError(f"prior_mean must have shape ({dim},), got {tuple(mu0.shape)}")

        # Prior precision: Lambda0 = (1/prior_var) I
        self._Lambda = (1.0 / self.config.prior_var) * torch.eye(
            dim, device=self.device, dtype=self.dtype
        )
        # b0 = Lambda0 * mu0
        self._b = self._Lambda @ mu0

        # Cached posterior params (invalidated on update)
        self._mu_cache: Optional[torch.Tensor] = None
        self._chol_cache: Optional[torch.Tensor] = None

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def prior_var(self) -> float:
        return self.config.prior_var

    @property
    def obs_noise_var(self) -> float:
        return self.config.obs_noise_var

    def precision_matrix(self) -> torch.Tensor:
        """Return current posterior precision Lambda."""
        return self._Lambda

    def _invalidate_cache(self) -> None:
        self._mu_cache = None
        self._chol_cache = None

    def posterior_mean(self) -> torch.Tensor:
        """
        Return posterior mean mu (cached).

        Why cache self._mu_cache?
            posterior_mean() may be called many times (e.g., scoring many actions).
            Solving a linear system repeatedly is expensive, so we cache until the
            next update() invalidates it.

        PSD vs SPD:
            - PSD = positive semidefinite: v^T A v >= 0 (may be singular)
            - SPD = symmetric positive definite: v^T A v > 0 (invertible)

        Invertibility of Lambda:
            self._Lambda starts as (1/prior_var) * I, which is SPD (prior_var > 0).
            Each update adds (Z^T Z) / obs_noise_var, which is PSD.
            SPD + PSD is SPD, so self._Lambda remains SPD and therefore invertible
            in exact arithmetic.

        Note:
            For sampling we add `jitter * I` before Cholesky to improve numerical
            stability under floating-point roundoff.
        """
        if self._mu_cache is None:
            # Solve Lambda * mu = b
            self._mu_cache = torch.linalg.solve(self._Lambda, self._b)
        return self._mu_cache

    def _chol_precision(self) -> torch.Tensor:
        """
        Cholesky factor L of (Lambda + jitter*I), where Lambda = L L^T.
        Cached for sampling stability.
        """
        if self._chol_cache is None:
            jitter_I = self.config.jitter * torch.eye(self.dim, device=self.device, dtype=self.dtype)
            self._chol_cache = torch.linalg.cholesky(self._Lambda + jitter_I)
        return self._chol_cache

    def update(self, z: torch.Tensor, r: torch.Tensor | float) -> None:
        """
        Update posterior with one or a batch of observations.

        Args:
            z: feature vector(s), shape (dim,) or (n, dim)
            r: reward(s), scalar or shape (n,)
        """
        Z = z.to(device=self.device, dtype=self.dtype)
        if Z.ndim == 1:
            Z = Z.unsqueeze(0)
        if Z.ndim != 2 or Z.shape[1] != self.dim:
            raise ValueError(f"z must have shape (dim,) or (n, dim), got {tuple(Z.shape)}")

        if isinstance(r, float) or (isinstance(r, torch.Tensor) and r.ndim == 0):
            r_t = torch.tensor([float(r)], device=self.device, dtype=self.dtype)
        else:
            r_t = r.to(device=self.device, dtype=self.dtype)
            if r_t.ndim != 1:
                raise ValueError(f"r must be scalar or shape (n,), got {tuple(r_t.shape)}")
            if r_t.shape[0] != Z.shape[0]:
                raise ValueError(f"r length {r_t.shape[0]} must match Z rows {Z.shape[0]}")

        inv_noise = 1.0 / self.config.obs_noise_var

        # Lambda += Z^T Z / noise_var
        self._Lambda = self._Lambda + inv_noise * (Z.T @ Z)
        # b += Z^T r / obs_noise_var
        self._b = self._b + inv_noise * (Z.T @ r_t)

        self._invalidate_cache()

    def sample_weights(
        self,
        *,
        n_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample weights from posterior w ~ N(mu, Sigma), where Sigma = Lambda^{-1}.

        Uses precision Cholesky: Lambda = L L^T.
        If eps ~ N(0, I), then delta_w = L^{-T} eps has covariance Lambda^{-1}.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        mu = self.posterior_mean()
        L = self._chol_precision()

        eps = torch.randn(
            (n_samples, self.dim), device=self.device, dtype=self.dtype, generator=generator
        )
        # Solve L^T delta_w = eps^T  => delta_w = L^{-T} eps
        delta_w = torch.linalg.solve_triangular(L.T, eps.T, upper=True).T

        w = mu.unsqueeze(0) + delta_w
        return w[0] if n_samples == 1 else w

    def mean_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return the posterior-mean score for a feature vector.

        This is the deterministic prediction under the posterior mean weights:
            score = z^T mu

        Args:
            z: feature vector, shape (dim,)

        Returns:
            Scalar tensor equal to z^T mu.
        """
        z_t = z.to(device=self.device, dtype=self.dtype)
        if z_t.shape != (self.dim,):
            raise ValueError(f"z must have shape ({self.dim},), got {tuple(z_t.shape)}")
        return z_t @ self.posterior_mean()

    def sample_score(
        self,
        z: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Return a Thompson-sampled score for a feature vector.

        Samples weights w ~ N(mu, Sigma) and returns:
            score = z^T w

        Args:
            z: feature vector, shape (dim,)
            generator: optional torch RNG generator (for reproducibility)

        Returns:
            Scalar tensor equal to z^T w for a sampled w.
        """
        z_t = z.to(device=self.device, dtype=self.dtype)
        if z_t.shape != (self.dim,):
            raise ValueError(f"z must have shape ({self.dim},), got {tuple(z_t.shape)}")
        w = self.sample_weights(n_samples=1, generator=generator)
        return z_t @ w
    
    def state_dict(self) -> dict[str, object]:
        """
        Inference checkpoint state.

        We store CPU tensors so checkpoints are device-agnostic.
        """
        return {
            "dim": int(self.dim),
            "prior_var": float(self.config.prior_var),
            "obs_noise_var": float(self.config.obs_noise_var),
            "jitter": float(self.config.jitter),
            "_Lambda": self._Lambda.detach().cpu(),
            "_b": self._b.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """
        Load an inference checkpoint state into this instance.

        Assumes this BLR instance was constructed with the intended device/dtype.
        """
        if int(state["dim"]) != self.dim:
            raise ValueError(f"BLR dim mismatch: checkpoint {state['dim']} vs instance {self.dim}")

        Lambda = state["_Lambda"]
        b = state["_b"]
        if not isinstance(Lambda, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Invalid BLR checkpoint: _Lambda/_b must be torch.Tensors")

        if Lambda.shape != (self.dim, self.dim):
            raise ValueError(f"_Lambda must have shape ({self.dim},{self.dim}), got {tuple(Lambda.shape)}")
        if b.shape != (self.dim,):
            raise ValueError(f"_b must have shape ({self.dim},), got {tuple(b.shape)}")

        # Sanity-check config knobs (helps catch loading the wrong checkpoint)
        if abs(float(state["prior_var"]) - self.config.prior_var) > 1e-12:
            raise ValueError("BLR prior_var mismatch vs checkpoint")
        if abs(float(state["obs_noise_var"]) - self.config.obs_noise_var) > 1e-12:
            raise ValueError("BLR obs_noise_var mismatch vs checkpoint")
        if abs(float(state["jitter"]) - self.config.jitter) > 1e-12:
            raise ValueError("BLR jitter mismatch vs checkpoint")

        self._Lambda = Lambda.to(device=self.device, dtype=self.dtype)
        self._b = b.to(device=self.device, dtype=self.dtype)
        self._invalidate_cache()

