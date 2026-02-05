from __future__ import annotations

from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import torch
from torch import nn

from bandexa.buffers.base import (
    BufferConfig,
    MemoryReplayConfig,
    ReplayBufferProtocol,
    make_replay_buffer,
)
from bandexa.encoders.protocols import Encoder, has_encode_batch
from bandexa.posterior.bayes_linear import BayesianLinearRegression
from bandexa.posterior.bayes_logistic import BayesianLogisticRegression

Tensor = torch.Tensor
PosteriorType = Literal["bayes_linear", "bayes_logistic"]

# Loss override type: can be "mse"/"bce"/"auto", an nn.Module, or a callable loss.
LossSpec = Union[str, nn.Module, Callable[[Tensor, Tensor], Tensor], None]


@dataclass(frozen=True)
class NeuralLinearTSConfig:
    """
    Configuration for NeuralLinearTS.

    Design goals
    ------------
    We prefer configuration-driven construction so that in real deployments you can:
      - serialize configs to JSON/YAML,
      - pass them via CLI/service configs,
      - and evolve the internals (buffers/posteriors) without changing user code.

    Notes
    -----
    - `buffer` is a BufferConfig (memory/disk). The policy is intentionally buffer-backend agnostic.
    - `encode_kwargs` are default kwargs passed to encoder.encode / encoder.encode_batch.
      You can override per-call via encode_kwargs=... in select_action/update/train/rebuild.
    - `action_chunk_size` is the default chunk size for select_action (candidate scoring).
      This helps avoid encoding/scoring huge candidate sets in one go.
    """
    feature_dim: int

    # Posterior selection and hyperparameters
    posterior_type: PosteriorType = "bayes_linear"
    prior_var: float = 1.0
    obs_noise_var: float = 1.0  # only used by bayes_linear

    # Optimizer hyperparameters for encoder training
    lr: float = 1e-3

    # Replay buffer configuration (memory/disk/etc.)
    buffer: BufferConfig = field(default_factory=MemoryReplayConfig)

    # Serving-time action scoring control
    action_chunk_size: Optional[int] = None

    # Default encoder kwargs (merged with per-call overrides)
    encode_kwargs: dict[str, object] = field(default_factory=dict)


class NeuralLinearTS:
    """
    Neural Linear Thompson Sampling (NeuralLinear / NeuralTS).

    Key operational assumption:
      - `actions` passed to select_action() are the *candidate set* for this decision.
        If you have a massive action space, do candidate generation upstream and pass
        only a subset here.

    Encoder contract:
      - required: encoder.encode(context, action, **kwargs) -> z of shape (d,)
      - optional: encoder.encode_batch(context, actions, **kwargs) -> Z of shape (n_actions, d)

    Notes:
      - Posterior is updated online from (z_t, r_t)
      - Encoder can be trained periodically using replay buffer
      - After encoder training, you typically rebuild the posterior so it matches the new embeddings
      - select_action supports chunking to avoid encoding/scoring all candidates in one go

    Buffer backends:
      - memory: in-memory replay (fast, but bounded by RAM / GPU RAM)
      - disk:   disk-backed replay (sharded storage on disk; batch sampling without loading all data into RAM)

    The disk backend is the first step toward realistic production patterns where serving (action selection)
    and learning (posterior rebuild + encoder training) can be decoupled by writing experience to disk/object store
    and consuming it later in a training job/process.

    Clean construction (no backward compatibility)
    ----------------------------------------------
    This class is configuration-driven. Users should pass a NeuralLinearTSConfig, which in turn contains
    a BufferConfig. This keeps the policy agnostic to buffer types and makes scaling-oriented refactors
    (hybrid buffers, sharded buffers, streaming consumers) possible without editing this file again.
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        config: NeuralLinearTSConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        buffer: Optional[ReplayBufferProtocol] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        # --- Validate config early (fail fast) ---
        if config.feature_dim <= 0:
            raise ValueError(f"config.feature_dim must be positive, got {config.feature_dim}")
        if config.action_chunk_size is not None and config.action_chunk_size <= 0:
            raise ValueError(
                f"config.action_chunk_size must be positive or None, got {config.action_chunk_size}"
            )
        if config.prior_var <= 0:
            raise ValueError(f"config.prior_var must be > 0, got {config.prior_var}")
        if config.posterior_type == "bayes_linear" and config.obs_noise_var <= 0:
            raise ValueError(f"config.obs_noise_var must be > 0 for bayes_linear, got {config.obs_noise_var}")
        if config.lr <= 0:
            raise ValueError(f"config.lr must be > 0, got {config.lr}")

        self.config = config
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # Keep runtime copies of (possibly overridden by checkpoints) metadata.
        # These originate from config but can be replaced by load_inference().
        self._encode_kwargs: dict[str, object] = dict(config.encode_kwargs)
        self._action_chunk_size: Optional[int] = config.action_chunk_size

        self.encoder = encoder
        if isinstance(self.encoder, nn.Module):
            self.encoder.to(device=self.device, dtype=self.dtype)

        # Store ctor kwargs so rebuild_posterior can re-create the posterior cleanly.
        # bayes_linear uses obs_noise_var; bayes_logistic (stub) may not.
        self._posterior_kwargs_linear = dict(
            dim=int(self.config.feature_dim),
            prior_var=float(self.config.prior_var),
            obs_noise_var=float(self.config.obs_noise_var),
            dtype=self.dtype,
            device=self.device,
        )
        self._posterior_kwargs_logistic = dict(
            dim=int(self.config.feature_dim),
            prior_var=float(self.config.prior_var),
            dtype=self.dtype,
            device=self.device,
        )

        # Build posterior ("policy core") and pick a consistent encoder-training loss.
        self.posterior = self._build_posterior(self.config.posterior_type)
        self._loss_name, self._loss_fn = self._loss_for_posterior(self.config.posterior_type)

        # Replay buffer: either use provided instance, or build from BufferConfig via factory.
        # The policy remains buffer-backend agnostic.
        self.buffer: ReplayBufferProtocol
        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = make_replay_buffer(self.config.buffer, device=self.device, dtype=self.dtype)

        # Optimizer: either user-provided, or create default Adam over encoder parameters.
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if not isinstance(self.encoder, nn.Module):
                raise ValueError("If encoder is not a torch.nn.Module, you must pass an optimizer explicitly.")
            params = list(self.encoder.parameters())
            if len(params) == 0:
                raise ValueError("Encoder has no parameters; pass optimizer explicitly or provide a trainable encoder.")
            self.optimizer = torch.optim.Adam(params, lr=float(self.config.lr))

        self._train_step = 0  # monotonically increasing training-step counter

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    @property
    def train_loss_name(self) -> str:
        """Loss used for encoder training, derived from posterior_type (unless overridden)."""
        return self._loss_name

    def flush_buffer(self) -> None:
        """
        Flush buffered experience to durable storage, if the underlying buffer supports it.

        - DiskReplayBuffer: commonly writes the current in-progress shard to disk
        - In-memory buffer: typically no-op (does not have flush)
        """
        if hasattr(self.buffer, "flush"):
            self.buffer.flush()  # type: ignore[call-arg]

    def _merge_encode_kwargs(self, encode_kwargs: Optional[dict[str, object]]) -> dict[str, object]:
        if encode_kwargs is None:
            return self._encode_kwargs
        merged = dict(self._encode_kwargs)
        merged.update(encode_kwargs)
        return merged

    def _build_posterior(self, posterior_type: PosteriorType) -> object:
        """
        Construct the posterior object based on posterior_type.

        - bayes_linear   -> BayesianLinearRegression (Gaussian likelihood)
        - bayes_logistic -> BayesianLogisticRegression (Bernoulli likelihood; currently a stub)
        """
        if posterior_type == "bayes_linear":
            return BayesianLinearRegression(**self._posterior_kwargs_linear)

        if posterior_type == "bayes_logistic":
            # Be tolerant to minor signature differences in the stub.
            try:
                return BayesianLogisticRegression(**self._posterior_kwargs_logistic)
            except TypeError:
                return BayesianLogisticRegression(
                    dim=self._posterior_kwargs_logistic["dim"],
                    prior_var=self._posterior_kwargs_logistic["prior_var"],
                )

        raise ValueError(f"Unknown posterior_type '{posterior_type}'")

    def _loss_for_posterior(self, posterior_type: PosteriorType) -> tuple[str, nn.Module]:
        """
        Choose a supervised loss for training the encoder, consistent with posterior likelihood.

        - bayes_linear   -> MSE (Gaussian)
        - bayes_logistic -> BCEWithLogits (Bernoulli)
        """
        if posterior_type == "bayes_linear":
            return "mse", nn.MSELoss()
        if posterior_type == "bayes_logistic":
            return "bce", nn.BCEWithLogitsLoss()
        raise ValueError(f"Unknown posterior_type '{posterior_type}'")

    def _resolve_loss(self, loss: LossSpec) -> tuple[str, Union[nn.Module, Callable[[Tensor, Tensor], Tensor]]]:
        """
        Resolve a loss specification into a callable loss function.

        This supports tests and real-world usage where you may want to override the default loss:
          - loss=None or loss="auto": use the default loss derived from posterior_type
          - loss="mse" / "bce": choose a built-in loss
          - loss=nn.Module: any PyTorch loss module, e.g. nn.SmoothL1Loss()
          - loss=callable(pred, target): a custom function returning a tensor loss

        Note:
          - For bayes_logistic, the default uses BCEWithLogitsLoss, and pred is treated as logits.
          - For bayes_linear, the default uses MSELoss.
        """
        if loss is None or loss == "auto":
            return self._loss_name, self._loss_fn

        if isinstance(loss, str):
            key = loss.lower()
            if key == "mse":
                return "mse", nn.MSELoss()
            if key in ("bce", "bcewithlogits", "bce_with_logits"):
                return "bce", nn.BCEWithLogitsLoss()
            raise ValueError(f"Unknown loss string '{loss}'. Supported: 'auto', 'mse', 'bce'.")

        if isinstance(loss, nn.Module):
            return loss.__class__.__name__, loss

        if callable(loss):
            return "callable", loss

        raise TypeError(f"Invalid loss spec type: {type(loss)}")

    def _batch_to_device_dtype(self, batch: object) -> tuple[Tensor, Tensor, Tensor]:
        """
        Normalize a batch to (contexts, actions, rewards) tensors on (self.device, self.dtype).

        Disk-backed buffers often store on CPU and may return CPU tensors from sample/iter_batches.
        This method makes the policy robust: it always trains/encodes on the policy device/dtype.

        Expected batch attributes (by protocol):
          - batch.contexts: Tensor
          - batch.actions:  Tensor
          - batch.rewards:  Tensor
        """
        if not hasattr(batch, "contexts") or not hasattr(batch, "actions") or not hasattr(batch, "rewards"):
            raise TypeError("Batch object must have .contexts, .actions, .rewards attributes")

        ctx = getattr(batch, "contexts")
        act = getattr(batch, "actions")
        rew = getattr(batch, "rewards")

        if not isinstance(ctx, torch.Tensor) or not isinstance(act, torch.Tensor) or not isinstance(rew, torch.Tensor):
            raise TypeError("Batch attributes .contexts/.actions/.rewards must be torch.Tensor")

        ctx = ctx.to(device=self.device, dtype=self.dtype)
        act = act.to(device=self.device, dtype=self.dtype)

        # Rewards are treated as float targets for both MSE and BCE-with-logits.
        # Buffers may store rewards as float32 already; if not, cast.
        rew = rew.to(device=self.device, dtype=self.dtype)

        # Common shapes: (B,), (B,1). Normalize to (B,) for loss convenience.
        if rew.ndim == 2 and rew.shape[1] == 1:
            rew = rew.squeeze(1)
        elif rew.ndim != 1:
            rew = rew.reshape(-1)

        return ctx, act, rew

    def select_action(
        self,
        context: Tensor,
        actions: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        encode_kwargs: Optional[dict[str, object]] = None,
        chunk_size: Optional[int] = None,
    ) -> int:
        """
        Choose an action index via Thompson sampling over the *provided candidate actions*.

        Args:
            context: context tensor (shape user-defined)
            actions: tensor of candidate actions, typically (n_actions, *action_shape)
            generator: optional RNG generator
            encode_kwargs: optional kwargs passed through to encoder.encode / encode_batch
            chunk_size: if provided, encode/score actions in chunks of this size.
                        If None, uses config.action_chunk_size.

        Returns:
            index of selected action in [0, n_actions)
        """
        if actions.ndim == 0:
            raise ValueError("actions must be at least 1D (n_actions, ...)")

        ctx = context.to(device=self.device, dtype=self.dtype)
        acts = actions.to(device=self.device, dtype=self.dtype)
        kw = self._merge_encode_kwargs(encode_kwargs)

        n_actions = int(acts.shape[0])
        if n_actions == 0:
            raise ValueError("actions must contain at least one candidate action")

        cs = self._action_chunk_size if chunk_size is None else chunk_size
        if cs is not None and cs <= 0:
            raise ValueError(f"chunk_size must be positive or None, got {cs}")
        if cs is None:
            cs = n_actions  # no chunking

        if not hasattr(self.posterior, "sample_weights"):
            raise TypeError("posterior must implement sample_weights()")

        # Sample once per decision (TS)
        w = self.posterior.sample_weights(n_samples=1, generator=generator)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] == 1:
            w = w[0]
        if not isinstance(w, torch.Tensor) or w.shape != (self.feature_dim,):
            raise ValueError(
                f"sample_weights must return shape ({self.feature_dim},), got {getattr(w, 'shape', None)}"
            )

        best_score: Optional[float] = None
        best_idx = 0

        with torch.no_grad():
            if has_encode_batch(self.encoder):
                for start in range(0, n_actions, cs):
                    end = min(start + cs, n_actions)
                    Z = self.encoder.encode_batch(ctx, acts[start:end], **kw)
                    if Z.ndim != 2 or Z.shape[1] != self.feature_dim:
                        raise ValueError(
                            f"encoder.encode_batch must return Z with shape (k, {self.feature_dim}), got {tuple(Z.shape)}"
                        )
                    scores = Z @ w
                    local_best = int(torch.argmax(scores).item())
                    local_best_score = float(scores[local_best].item())

                    if best_score is None or local_best_score > best_score:
                        best_score = local_best_score
                        best_idx = start + local_best
            else:
                for start in range(0, n_actions, cs):
                    end = min(start + cs, n_actions)
                    zs = [self.encoder.encode(ctx, acts[i], **kw) for i in range(start, end)]
                    Z = torch.stack(zs, dim=0)
                    if Z.ndim != 2 or Z.shape[1] != self.feature_dim:
                        raise ValueError(
                            f"encoder.encode must return z with shape ({self.feature_dim},) for each action; "
                            f"stacked Z expected (k, {self.feature_dim}), got {tuple(Z.shape)}"
                        )
                    scores = Z @ w
                    local_best = int(torch.argmax(scores).item())
                    local_best_score = float(scores[local_best].item())

                    if best_score is None or local_best_score > best_score:
                        best_score = local_best_score
                        best_idx = start + local_best

        return int(best_idx)

    def update(
        self,
        context: Tensor,
        action: Tensor,
        reward: float | Tensor,
        *,
        encode_kwargs: Optional[dict[str, object]] = None,
    ) -> None:
        """
        Update posterior and log the experience into the replay buffer.

        Args:
            context: context tensor
            action: chosen action tensor
            reward: observed reward (float or scalar tensor)
            encode_kwargs: optional kwargs passed through to encoder.encode
        """
        ctx = context.to(device=self.device, dtype=self.dtype)
        act = action.to(device=self.device, dtype=self.dtype)
        kw = self._merge_encode_kwargs(encode_kwargs)

        with torch.no_grad():
            z = self.encoder.encode(ctx, act, **kw)

        if z.shape != (self.feature_dim,):
            raise ValueError(f"encoder.encode must return shape ({self.feature_dim},), got {tuple(z.shape)}")

        if not hasattr(self.posterior, "update"):
            raise TypeError("posterior must implement update()")

        self.posterior.update(z, reward)

        # Buffer implementations decide how/where to store:
        # - memory: store tensors on configured device
        # - disk:   cast/store on CPU and persist to disk in shards
        self.buffer.add(ctx, act, reward)

    def train_encoder(
        self,
        *,
        optimizer_steps: int = 50,
        batch_size: int = 64,
        loss: LossSpec = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, dict[str, object]], None]] = None,
        log_every: int = 10,
        encode_kwargs: Optional[dict[str, object]] = None,
    ) -> float:
        """
        Train encoder parameters using logged tuples (context, action, reward).

        Loss is selected automatically from posterior_type unless overridden:

          - bayes_linear   -> MSE
          - bayes_logistic -> BCEWithLogits

        You can override via `loss=`:
          - loss="mse" / "bce" / "auto"
          - loss=nn.Module (e.g., nn.SmoothL1Loss())
          - loss=callable(pred, target) -> tensor
        """
        if optimizer_steps <= 0:
            raise ValueError(f"optimizer_steps must be positive, got {optimizer_steps}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")
        if len(self.buffer) == 0:
            raise ValueError("buffer is empty; cannot train encoder")

        if not isinstance(self.encoder, nn.Module):
            raise ValueError("Encoder is not a torch.nn.Module; cannot train without parameters/optimizer")

        if not hasattr(self.posterior, "posterior_mean"):
            raise TypeError("posterior must implement posterior_mean()")

        kw = self._merge_encode_kwargs(encode_kwargs)
        loss_name, loss_fn = self._resolve_loss(loss)

        self.encoder.train()
        last_loss = 0.0

        for i in range(int(optimizer_steps)):
            batch = self.buffer.sample(int(batch_size), generator=generator)
            ctx_b, act_b, rew_b = self._batch_to_device_dtype(batch)

            Z = torch.stack(
                [self.encoder.encode(ctx_b[j], act_b[j], **kw) for j in range(rew_b.shape[0])],
                dim=0,
            )  # (B, d)

            if Z.ndim != 2 or Z.shape[1] != self.feature_dim:
                raise ValueError(f"encoder must return Z with shape (B, {self.feature_dim}), got {tuple(Z.shape)}")

            mu = self.posterior.posterior_mean().detach()
            if not isinstance(mu, torch.Tensor) or mu.shape != (self.feature_dim,):
                raise ValueError(
                    f"posterior_mean must return shape ({self.feature_dim},), got {getattr(mu, 'shape', None)}"
                )

            pred = Z @ mu  # (B,)  (logits under bayes_logistic)

            # nn.Module or callable loss are both supported here
            loss_t = loss_fn(pred, rew_b)  # type: ignore[misc]
            if isinstance(loss_t, torch.Tensor) and loss_t.ndim != 0:
                loss_t = loss_t.mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss_t.backward()
            self.optimizer.step()

            last_loss = float(loss_t.detach().cpu().item())
            self._train_step += 1

            if callback is not None and (i % log_every == 0 or i == optimizer_steps - 1):
                callback(
                    self._train_step,
                    {
                        "loss": last_loss,
                        "loss_name": loss_name,
                        "buffer_size": float(len(self.buffer)),
                    },
                )

        return last_loss

    def rebuild_posterior(
        self,
        *,
        encode_kwargs: Optional[dict[str, object]] = None,
        chunk_size: int = 4096,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Rebuild/reset the posterior from the replay buffer using the *current* encoder.

        Streams through the buffer in chunks to avoid allocating a huge (N, d) embedding matrix.

        Why iter_batches is required
        ----------------------------
        For disk-backed replay buffers, materializing the entire dataset into RAM is not an option.
        We therefore standardize on an iterator-style API:
            buffer.iter_batches(batch_size=..., shuffle=..., generator=...)
        Memory buffers can implement this trivially; disk buffers can stream from storage.

        Important:
          - We flush the buffer first if flush() exists (e.g. to persist the last partially filled shard).
          - We set the encoder to eval() during rebuild to make embeddings deterministic for modules
            with dropout/batchnorm (common in pretrained backbones).
        """
        if len(self.buffer) == 0:
            raise ValueError("buffer is empty; cannot rebuild posterior")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # Ensure any buffered-but-not-durable data is included (disk buffers commonly benefit from this).
        self.flush_buffer()

        kw = self._merge_encode_kwargs(encode_kwargs)

        # Reset posterior to its prior (same type as config)
        self.posterior = self._build_posterior(self.config.posterior_type)
        self._loss_name, self._loss_fn = self._loss_for_posterior(self.config.posterior_type)

        if not hasattr(self.posterior, "update"):
            raise TypeError("posterior must implement update()")

        # Put encoder in eval mode for deterministic embeddings (if it's an nn.Module).
        if isinstance(self.encoder, nn.Module):
            self.encoder.eval()

        with torch.no_grad():
            for batch in self.buffer.iter_batches(
                batch_size=int(chunk_size),
                shuffle=bool(shuffle),
                generator=generator,
            ):
                ctx_b, act_b, rew_b = self._batch_to_device_dtype(batch)

                Z_chunk = torch.stack(
                    [self.encoder.encode(ctx_b[i], act_b[i], **kw) for i in range(rew_b.shape[0])],
                    dim=0,
                )

                if Z_chunk.ndim != 2 or Z_chunk.shape[1] != self.feature_dim:
                    raise ValueError(
                        f"encoder.encode must return z with shape ({self.feature_dim},); "
                        f"stacked Z_chunk expected (k, {self.feature_dim}), got {tuple(Z_chunk.shape)}"
                    )

                self.posterior.update(Z_chunk, rew_b)

    def save_inference(self, path: str | PathLike[str]) -> None:
        """
        Save an inference-only checkpoint.

        Includes:
          - encoder weights (saved on CPU)
          - posterior state
          - minimal metadata (feature_dim, posterior_type, encode_kwargs, action_chunk_size, posterior hyperparams)

        Excludes:
          - replay buffer (memory/disk)
          - optimizer
        """
        if not hasattr(self.encoder, "state_dict"):
            raise TypeError("Encoder has no state_dict(); inference checkpointing requires nn.Module-like encoder")
        if not hasattr(self.posterior, "state_dict"):
            raise TypeError("Posterior has no state_dict(); cannot save inference checkpoint")

        enc_state = self.encoder.state_dict()
        enc_state_cpu = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in enc_state.items()
        }

        ckpt: dict[str, object] = {
            "version": 2,
            "feature_dim": int(self.feature_dim),
            "posterior_type": self.config.posterior_type,
            "prior_var": float(self.config.prior_var),
            "obs_noise_var": float(self.config.obs_noise_var),
            "encode_kwargs": dict(self._encode_kwargs),
            "action_chunk_size": self._action_chunk_size,
            "encoder_state": enc_state_cpu,
            "posterior_state": self.posterior.state_dict(),
        }
        torch.save(ckpt, str(Path(path)))

    def load_inference(
        self,
        path: str | PathLike[str],
        *,
        map_location: str | torch.device | None = None,
    ) -> None:
        """
        Load an inference-only checkpoint into this (already-constructed) instance.

        Replaces:
          - encoder weights
          - posterior type + posterior state
          - encode_kwargs + action_chunk_size
          - posterior hyperparams (prior_var, obs_noise_var) used to rebuild posterior object before loading state

        Does NOT load:
          - replay buffer
          - optimizer state

        Important:
          - This assumes the encoder architecture matches the checkpoint (state_dict-compatible).
          - feature_dim must match (since it is the posterior dimension and the encoder output dimension).
        """
        ckpt = torch.load(str(Path(path)), map_location=map_location)

        feat = int(ckpt.get("feature_dim"))
        if feat != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: checkpoint {feat} vs instance {self.feature_dim}")

        posterior_type = ckpt.get("posterior_type", "bayes_linear")
        if posterior_type not in ("bayes_linear", "bayes_logistic"):
            raise ValueError(f"Invalid posterior_type in checkpoint: {posterior_type}")

        prior_var = float(ckpt.get("prior_var", self.config.prior_var))
        obs_noise_var = float(ckpt.get("obs_noise_var", self.config.obs_noise_var))

        # Update config in a single immutable replace (buffer/lr/etc. remain as constructed).
        self.config = replace(
            self.config,
            posterior_type=posterior_type,  # type: ignore[arg-type]
            prior_var=prior_var,
            obs_noise_var=obs_noise_var,
        )

        # Refresh posterior ctor kwargs
        self._posterior_kwargs_linear.update(
            prior_var=float(self.config.prior_var),
            obs_noise_var=float(self.config.obs_noise_var),
        )
        self._posterior_kwargs_logistic.update(prior_var=float(self.config.prior_var))

        # Encoder weights
        enc_state = ckpt.get("encoder_state")
        if enc_state is None:
            raise ValueError("Checkpoint missing encoder_state")
        if not hasattr(self.encoder, "load_state_dict"):
            raise TypeError("Encoder has no load_state_dict()")
        self.encoder.load_state_dict(enc_state)

        # Rebuild posterior object for checkpoint type and load its state
        self.posterior = self._build_posterior(self.config.posterior_type)
        self._loss_name, self._loss_fn = self._loss_for_posterior(self.config.posterior_type)

        post_state = ckpt.get("posterior_state")
        if post_state is None:
            raise ValueError("Checkpoint missing posterior_state")
        if not hasattr(self.posterior, "load_state_dict"):
            raise TypeError("Posterior has no load_state_dict()")
        self.posterior.load_state_dict(post_state)

        # metadata
        self._encode_kwargs = dict(ckpt.get("encode_kwargs", {}))
        self._action_chunk_size = ckpt.get("action_chunk_size", None)
