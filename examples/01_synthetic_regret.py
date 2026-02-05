# examples/01_synthetic_regret.py
"""
Synthetic contextual bandit example with regret.

Compares:
  1) LinearTS: fixed features phi(x,a) = concat(x,a) with a Bayesian linear posterior
  2) NeuralLinearTS: learned features z = encoder(conbine(x,a)) with a Bayesian posterior
     (posterior_type="bayes_linear" in this example)

We generate a nonlinear reward function (with interaction terms) so that
a learned encoder can outperform a purely linear model on raw concat features.

Run:
  python examples/01_synthetic_regret.py
  python examples/01_synthetic_regret.py --help
  python examples/01_synthetic_regret.py --device cuda
  python examples/01_synthetic_regret.py --plot

  # make it more nonlinear (usually increases NeuralTS advantage)
  python examples/01_synthetic_regret.py --interaction-scale 1.5

  # sanity-check save/load at the end of the run
  python examples/01_synthetic_regret.py --sanity-check-save-load

Notes:
  - Regret is computed w.r.t. the expected reward under the synthetic environment.
  - save/load is "inference-only" (encoder + posterior), no replay buffer or optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from bandexa.buffers.base import MemoryReplayConfig
from bandexa.posterior.bayes_linear import BayesianLinearRegression
from bandexa.policies.neural_linear_ts import NeuralLinearTS, NeuralLinearTSConfig


# -------------------------
# Synthetic environment
# -------------------------


@dataclass(frozen=True)
class SyntheticEnvConfig:
    dim: int = 16
    n_actions: int = 50
    env_noise_std: float = 0.10
    seed: int = 0
    interaction_scale: float = 1.0  # scales the interaction term weights w_xa


class SyntheticEnv:
    """
    Simulated environment with a nonlinear expected reward, defined by the model:
        m(x,a) = tanh( w_x^T x + w_a^T a + w_xa^T (x ⊙ a) + b )

    The interaction term (x ⊙ a) is included to make the reward depend on interactions,
    such that a purely linear model on [x; a] struggles, while a neural encoder can learn
    features that capture those interactions better, so we can actually see NeuralLinearTS
    outperforms LinearTS.

    Observed reward:
        r = m(x,a) + Normal(0, env_noise_std^2)

    "actions" is the fixed action set (the arms) for the whole run: shape (n_actions, dim).
    Each action is represented by a feature vector a ∈ R^dim.
    We draw them from Normal(0,1) just to get a diverse, generic set of arms without designing
    anything special. Any distribution would work; Gaussian is a simple default.

    w_x, w_a, w_xa, and b are the hidden parameters of the environment's reward function.
    w_x ∈ R^dim: weights for the context-only effect
    w_a ∈ R^dim: weights for the action-only effect
    w_xa ∈ R^dim: weights for interaction between context and action
    b: a scalar bias

    interaction_scale scales the interaction weights w_xa. Larger values make the environment
    more nonlinear and typically increase the advantage of NeuralLinearTS over LinearTS.

    Regret is computed using the expected reward m(x,a).
    """

    def __init__(self, cfg: SyntheticEnvConfig, *, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        if cfg.interaction_scale <= 0:
            raise ValueError(f"interaction_scale must be > 0, got {cfg.interaction_scale}")

        # Use a CPU generator so env is reproducible across cpu/cuda runs.
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)

        # fixed global action set
        self.actions = torch.randn(cfg.n_actions, cfg.dim, generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

        # hidden parameters defining the environment
        self.w_x = torch.randn(cfg.dim, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        self.w_a = torch.randn(cfg.dim, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        self.w_xa = (
            cfg.interaction_scale
            * torch.randn(cfg.dim, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        )
        self.b = torch.randn((), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    def sample_context(self, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample x_t ~ N(0, I) in R^dim."""
        if generator is None:
            return torch.randn(self.cfg.dim, device=self.device, dtype=self.dtype)
        return torch.randn(self.cfg.dim, device=self.device, dtype=self.dtype, generator=generator)

    def mean_reward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Ground-truth expected reward m(x, a)."""
        return torch.tanh((x @ self.w_x) + (a @ self.w_a) + ((x * a) @ self.w_xa) + self.b)

    def observe_reward(self, mean: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add observation noise to the mean reward."""
        eps = (
            torch.randn((), device=self.device, dtype=self.dtype, generator=generator)
            if generator is not None
            else torch.randn((), device=self.device, dtype=self.dtype)
        )
        return mean + self.cfg.env_noise_std * eps

    def oracle_best(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Compute best expected reward among all actions for this context x.
        Returns (best_idx, best_mean).
        """
        means = torch.stack([self.mean_reward(x, a) for a in self.actions], dim=0)  # (n_actions,)
        best_idx = int(torch.argmax(means).item())
        return best_idx, means[best_idx]


# -------------------------
# Agents
# -------------------------


class LinearTSAgent:
    """
    Linear Thompson Sampling baseline with fixed features phi(x,a) = concat(x,a).

    Uses BayesianLinearRegression directly on phi(x,a).

    It doesn't implement encode() because it isn't using an encoder at all.
    It's the baseline “fixed feature” method where the feature map is
    explicitly phi(x,a)=concat(x,a), so it just has a phi() helper and
    feeds that into BayesianLinearRegression.
    """

    def __init__(
        self,
        *,
        dim: int,
        prior_var: float,
        obs_noise_var: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.dim = int(dim)
        self.phi_dim = 2 * self.dim
        self.device = device
        self.dtype = dtype

        self.posterior = BayesianLinearRegression(
            dim=self.phi_dim,
            prior_var=float(prior_var),
            obs_noise_var=float(obs_noise_var),
            device=device,
            dtype=dtype,
        )

    def phi(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Fixed feature map phi(x,a) = [x; a]."""
        return torch.cat([x, a], dim=-1)

    def select_action(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        w = self.posterior.sample_weights(n_samples=1, generator=generator)  # (phi_dim,)
        X = x.unsqueeze(0).expand(actions.shape[0], -1)
        Phi = torch.cat([X, actions], dim=-1)  # (n_actions, phi_dim)
        scores = Phi @ w
        return int(torch.argmax(scores).item())

    def update(self, x: torch.Tensor, a: torch.Tensor, r: float | torch.Tensor) -> None:
        z = self.phi(x, a)
        self.posterior.update(z, r)


class MLPConcatEncoder(nn.Module):
    """
    Trainable joint encoder:
      z = f([context; action])

    Implements "encode()" and "encode_batch()" to hit the fast-path in NeuralLinearTS.

    Relationship to Encoder protocol: MLPConcatEncoder defines the required
    methods with compatible signatures, it “is an Encoder” for type-checkers and
    for runtime usage.
    """

    def __init__(self, dim: int, feature_dim: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = 2 * int(dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(feature_dim)),
        )

    def encode(self, context: torch.Tensor, action: torch.Tensor, **kwargs: object) -> torch.Tensor:
        x = torch.cat([context, action], dim=-1)
        return self.net(x)

    def encode_batch(self, context: torch.Tensor, actions: torch.Tensor, **kwargs: object) -> torch.Tensor:
        ctx = context.unsqueeze(0).expand(actions.shape[0], -1)
        x = torch.cat([ctx, actions], dim=-1)
        return self.net(x)


# -------------------------
# Utilities
# -------------------------


def moving_avg(x: list[float], window: int) -> float:
    """
    computes the average of the most recent window values in the list x,
    or all of them if there are fewer than window, so we can print a
    smoother “recent reward” metric instead of noisy per-step rewards.
    """
    if not x:
        return 0.0
    w = min(window, len(x))
    return float(sum(x[-w:]) / w)


def maybe_plot(curves: dict[str, list[float]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; skipping plot")
        return

    plt.figure()
    for name, ys in curves.items():
        plt.plot(ys, label=name)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _probe_numbers(neural: NeuralLinearTS, x: torch.Tensor, actions: torch.Tensor, a_idx: int) -> tuple[float, float]:
    """
    Deterministic probe for save/load sanity:
      - mean_score for (x, chosen_action) under posterior mean
      - L2 norm of posterior mean vector mu
    """
    neural.encoder.eval()
    with torch.no_grad():
        z = neural.encoder.encode(x, actions[a_idx])  # (d,)
        # Use dot(z, mu) so we don't depend on a specific posterior API beyond posterior_mean().
        mu = neural.posterior.posterior_mean().detach()  # (d,)
        score = float((z @ mu).detach().cpu().item())
        mu_norm = float(mu.norm(p=2).detach().cpu().item())
    return score, mu_norm


# -------------------------
# Main
# -------------------------


def main() -> None:
    """
    Run a synthetic contextual bandit simulation comparing LinearTS vs NeuralLinearTS.

    Command-line arguments
    ----------------------
    --horizon : int (default: 2000)
        Number of online interaction steps to run. Each step samples a new context,
        selects an action, observes a reward, and updates the posterior (and optionally
        trains the encoder periodically).

    --dim : int (default: 16)
        Dimensionality of the context vector x and each action vector a in the synthetic
        environment (x ∈ R^dim, a ∈ R^dim).

    --actions : int (default: 50)
        Number of candidate actions (arms) available at each step. In this example the
        action set is fixed and reused across all steps.

    --seed : int (default: 0)
        Random seed used for reproducibility (environment parameters, action set, etc.).

    --env-noise-std : float (default: 0.10)
        Standard deviation of the observation noise added to the environment's mean reward.
        Observed reward is: r = mean_reward(x, a) + Normal(0, noise_std^2).

    --interaction-scale : float (default: 1.0)

    --blr-prior-var : float (default: 5.0)
        Prior variance for Bayesian linear regression weights w (isotropic prior):
        w ~ Normal(0, prior_var * I). Larger values imply a weaker prior (more uncertainty).

    --blr-noise-var : float or None (default: None)
        Observation noise variance parameter used in the Bayesian linear regression posterior
        update (BLR). If not provided, defaults to env_noise_std^2. This is the sigma^2 in the
        Gaussian likelihood used by the posterior math.

    --feature-dim : int (default: 32)
        Dimensionality of the learned feature vector z produced by the neural encoder in
        NeuralLinearTS (z ∈ R^feature_dim). The Bayesian posterior is maintained over weights
        w ∈ R^feature_dim.

    --train-every : int (default: 50)
        Frequency (in online steps) for training the neural encoder. Every train_every steps,
        the script runs train_steps gradient updates on minibatches from the replay buffer and
        then rebuilds the Bayesian posterior using the updated encoder.

    --optimizer-steps : int (default: 20)
        Number of gradient update steps to run during each periodic encoder-training phase.

    --batch-size : int (default: 64)
        Minibatch size sampled from the replay buffer for each encoder gradient update step.

    --rebuild-chunk : int (default: 1024)
        Chunk size used when rebuilding the Bayesian posterior from all logged data after encoder
        training. Smaller values reduce peak memory usage at the cost of more loop overhead.

    --action-chunk : int or None (default: None)
        Chunk size used inside NeuralLinearTS.select_action() when scoring many candidate actions.
        If set, actions are encoded/scored in chunks to reduce peak memory usage. If None, all
        candidates are encoded/scored in one batch.

    --device : {"cpu", "cuda"} (default: "cpu")
        Device to run the simulation on. If "cuda" is requested but not available, the script
        falls back to CPU. The neural encoder benefits most from GPU; the Bayesian posterior is
        typically small enough to run efficiently on CPU.

    --plot : flag (default: False)
        If set, plot regret curves at the end (requires matplotlib). If matplotlib is not installed,
        plotting is skipped gracefully.

    --save-path : str  (default: examples/data/neuralts_synth_inference.pt)
        Where to save an inference-only checkpoint at end of run (used by sanity check).

    --sanity-check-save-load : bool (default: False)
        If set, saves an inference checkpoint at t==horizon, reloads it into a fresh agent,
        and prints probe deltas to verify save/load correctness.
    """

    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=2000)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--actions", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-noise-std", type=float, default=0.10)
    p.add_argument("--interaction-scale", type=float, default=1.0)

    p.add_argument("--blr-prior-var", type=float, default=5.0)
    p.add_argument("--blr-noise-var", type=float, default=None, help="Posterior obs_noise_var; default = env_noise_std^2")

    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--train-every", type=int, default=50)
    p.add_argument("--optimizer-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rebuild-chunk", type=int, default=1024)
    p.add_argument("--action-chunk", type=int, default=None, help="Chunk candidates in select_action()")

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--plot", action="store_true")

    p.add_argument(
        "--save-path",
        type=str,
        default="examples/data/neuralts_synth_inference.pt",
        help="Where to save an inference-only checkpoint at end of run (used by sanity check).",
    )
    p.add_argument(
        "--sanity-check-save-load",
        action="store_true",
        help="If set, at the end of the run: compute a probe score, save inference checkpoint, "
        "reload into a fresh agent, and verify the probe score matches.",
    )

    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    blr_noise_var = float(args.env_noise_std**2) if args.blr_noise_var is None else float(args.blr_noise_var)

    # Global seeds for ops that don't accept generator=...
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Dedicated RNG streams (avoid coupling between env / agents / training)
    env_ctx_gen = torch.Generator(device=device).manual_seed(args.seed + 0)
    env_lin_noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    env_neu_noise_gen = torch.Generator(device=device).manual_seed(args.seed + 2)

    lin_ts_gen = torch.Generator(device=device).manual_seed(args.seed + 3)
    neu_ts_gen = torch.Generator(device=device).manual_seed(args.seed + 4)

    train_gen = torch.Generator(device=device).manual_seed(args.seed + 5)

    env_cfg = SyntheticEnvConfig(
        dim=int(args.dim),
        n_actions=int(args.actions),
        env_noise_std=float(args.env_noise_std),
        seed=int(args.seed),
        interaction_scale=float(args.interaction_scale),
    )
    env = SyntheticEnv(env_cfg, device=device, dtype=dtype)

    # Agents
    linear = LinearTSAgent(
        dim=int(args.dim),
        prior_var=float(args.blr_prior_var),
        obs_noise_var=blr_noise_var,
        device=device,
        dtype=dtype,
    )

    encoder = MLPConcatEncoder(dim=int(args.dim), feature_dim=int(args.feature_dim))

    # NeuralLinearTS now takes a config that includes the replay buffer backend config.
    # This example uses an in-memory replay buffer only.
    neural_cfg = NeuralLinearTSConfig(
        feature_dim=int(args.feature_dim),
        posterior_type="bayes_linear",
        prior_var=float(args.blr_prior_var),
        obs_noise_var=blr_noise_var,
        lr=1e-3,
        buffer=MemoryReplayConfig(capacity=50_000),
        action_chunk_size=args.action_chunk,
    )

    neural = NeuralLinearTS(
        encoder=encoder,
        config=neural_cfg,
        device=device,
        dtype=dtype,
    )

    # Metrics
    lin_rewards: list[float] = []
    neu_rewards: list[float] = []
    lin_regrets: list[float] = []
    neu_regrets: list[float] = []

    last_x: torch.Tensor | None = None
    last_a_idx_neu: int = 0

    horizon = int(args.horizon)

    for t in range(1, horizon + 1):
        x = env.sample_context(generator=env_ctx_gen).to(device=device, dtype=dtype)
        last_x = x

        # Oracle
        _, best_mean = env.oracle_best(x)

        # --- LinearTS ---
        a_idx_lin = linear.select_action(x, env.actions, generator=lin_ts_gen)
        a_lin = env.actions[a_idx_lin]
        mean_lin = env.mean_reward(x, a_lin)
        r_lin = env.observe_reward(mean_lin, generator=env_lin_noise_gen)

        linear.update(x, a_lin, r_lin)
        lin_rewards.append(float(r_lin.item()))
        lin_regrets.append(float((best_mean - mean_lin).item()))

        # --- NeuralLinearTS ---
        a_idx_neu = neural.select_action(x, env.actions, generator=neu_ts_gen)
        last_a_idx_neu = int(a_idx_neu)

        a_neu = env.actions[a_idx_neu]
        mean_neu = env.mean_reward(x, a_neu)
        r_neu = env.observe_reward(mean_neu, generator=env_neu_noise_gen)

        neural.update(x, a_neu, r_neu)
        neu_rewards.append(float(r_neu.item()))
        neu_regrets.append(float((best_mean - mean_neu).item()))

        # periodic encoder training + posterior rebuild
        if args.train_every > 0 and (t % int(args.train_every) == 0):
            if len(neural.buffer) >= max(8, int(args.batch_size)):
                _ = neural.train_encoder(
                    optimizer_steps=int(args.optimizer_steps),
                    batch_size=int(args.batch_size),
                    generator=train_gen,
                )
                neural.rebuild_posterior(chunk_size=int(args.rebuild_chunk))

        # progress
        if t in {10, 50, 100} or (t % 200 == 0) or (t == horizon):
            lin_cum_reg = float(sum(lin_regrets))
            neu_cum_reg = float(sum(neu_regrets))
            lin_ma = moving_avg(lin_rewards, window=200)
            neu_ma = moving_avg(neu_rewards, window=200)

            print(
                f"[t={t:4d}] "
                f"LinearTS  cum_reg={lin_cum_reg:8.2f}  ma_reward={lin_ma:+.3f} | "
                f"NeuralTS  cum_reg={neu_cum_reg:8.2f}  ma_reward={neu_ma:+.3f} | "
                f"device={device.type}"
            )

    # --- End-of-run save/load sanity check ---
    if args.sanity_check_save_load:
        if last_x is None:
            raise RuntimeError("Internal error: last_x is None after the loop.")
        ckpt_path = str(Path(args.save_path))
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        # Probe BEFORE save
        score_before, mu_norm_before = _probe_numbers(neural, last_x, env.actions, last_a_idx_neu)

        # Save inference checkpoint
        neural.save_inference(ckpt_path)

        # Rebuild a fresh agent with same architecture/config, then load
        encoder2 = MLPConcatEncoder(dim=int(args.dim), feature_dim=int(args.feature_dim))

        neural_cfg2 = NeuralLinearTSConfig(
            feature_dim=int(args.feature_dim),
            posterior_type="bayes_linear",
            prior_var=float(args.blr_prior_var),
            obs_noise_var=blr_noise_var,
            lr=1e-3,
            buffer=MemoryReplayConfig(capacity=50_000),
            action_chunk_size=args.action_chunk,
        )

        neural2 = NeuralLinearTS(
            encoder=encoder2,
            config=neural_cfg2,
            device=device,
            dtype=dtype,
        )
        neural2.load_inference(ckpt_path, map_location="cpu")

        # Probe AFTER load
        score_after, mu_norm_after = _probe_numbers(neural2, last_x, env.actions, last_a_idx_neu)

        print("\n[Sanity check: save/load inference]")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  probe action idx: {last_a_idx_neu}")
        print(
            f"  mean_score before: {score_before:+.8f}   after: {score_after:+.8f}   "
            f"delta: {score_after - score_before:+.3e}"
        )
        print(
            f"  ||mu||2   before: {mu_norm_before:+.8f}   after: {mu_norm_after:+.8f}   "
            f"delta: {mu_norm_after - mu_norm_before:+.3e}"
        )

        if abs(score_after - score_before) > 1e-7 or abs(mu_norm_after - mu_norm_before) > 1e-7:
            print("  WARNING: sanity check deltas are larger than expected (possible dtype/device mismatch).")
        else:
            print("  OK: save/load reproduced probe numbers.")

    # Final summary
    print("\nFinal:")
    print(f"  LinearTS cumulative regret: {sum(lin_regrets):.3f}")
    print(f"  NeuralTS cumulative regret: {sum(neu_regrets):.3f}")
    print(f"  LinearTS avg reward (last 200): {moving_avg(lin_rewards, 200):.3f}")
    print(f"  NeuralTS avg reward (last 200): {moving_avg(neu_rewards, 200):.3f}")

    if args.plot:
        maybe_plot({"LinearTS": lin_regrets, "NeuralTS": neu_regrets}, title="Instant regret (expected)")

        lin_cum: list[float] = []
        neu_cum: list[float] = []
        s1 = 0.0
        s2 = 0.0
        for r1, r2 in zip(lin_regrets, neu_regrets):
            s1 += r1
            s2 += r2
            lin_cum.append(s1)
            neu_cum.append(s2)
        maybe_plot({"LinearTS": lin_cum, "NeuralTS": neu_cum}, title="Cumulative regret")


if __name__ == "__main__":
    main()
