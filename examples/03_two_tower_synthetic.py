# examples/03_two_tower_synthetic.py
"""
Two-tower synthetic contextual bandit (large action set) example.

Goal
----
Show how Bandexa scales to large action sets without heavy datasets or torchvision:
  - a real two-tower encoder (context tower + action tower)
  - fast encode_batch() (compute context embedding once, score many actions)
  - candidate chunking in select_action() for big candidate sets
  - warmup exploration (uniform random), then Thompson Sampling

Environment
-----------
Synthetic *continuous* reward with *nonlinear* ground-truth structure.

We define a nonlinear mean reward in [0,1] using:

  u* = tanh(Wx x)
  v* = tanh(Wa a)
  logits(x,a) = <u*, v*> / sqrt(d) + 0.25 * mean((u* * v*)^2) + bias
  mean(x,a)   = sigmoid(logits(x,a))

Then we observe a *Gaussian* reward sample (not Bernoulli):

  reward ~ Normal(mean(x,a), sigma^2), and we clamp to [0,1] for interpretability.

Why Gaussian here?
- Our posterior is BayesianLinearRegression (Gaussian observation model).
- Using Gaussian rewards dramatically reduces the high-variance 0/1 noise from Bernoulli sampling,
  producing a cleaner, happier “demonstration” run.

Actions live in a large discrete set of size N (e.g., 10k), each represented by a fixed vector a ∈ R^{action_dim}.
Actions here are not one-hot. They're continuous action vectors taken from a fixed “action table”.
We create action_table with shape (N_actions, act_dim) (e.g. (10000, 32)).
At each step we either:
  - evaluate a candidate subset of size M (default), or
  - evaluate all N actions (if --candidate-size <= 0).

Agents
------
1) LinearTS baseline (fixed features):
   phi(x,a) = [x; a]  (purely linear in the raw inputs)

2) NeuralLinearTS (learned features):
   two-tower encoder with fusion:
     u = f_ctx(x)
     v = f_act(a)
     z = [u; v; u*v]
   Thompson sampling in feature space z via Bayesian linear regression posterior.

Run
---
  python examples/03_two_tower_synthetic.py
  python examples/03_two_tower_synthetic.py --help
  python examples/03_two_tower_synthetic.py --n-actions 10000 --candidate-size 1024
  python examples/03_two_tower_synthetic.py --device cuda --action-chunk 256

Notes
-----
- Pure synthetic tensors (no downloads, no torchvision).
- The point is mechanics: large action sets + encode_batch + chunking.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from bandexa.buffers.memory_replay import MemoryReplayBuffer
from bandexa.posterior.bayes_linear import BayesianLinearRegression
from bandexa.policies.neural_linear_ts import NeuralLinearTS, NeuralLinearTSConfig


# -------------------------
# Utilities
# -------------------------


def moving_avg(xs: list[float], window: int) -> float:
    if not xs:
        return 0.0
    w = min(int(window), len(xs))
    return float(sum(xs[-w:]) / w)


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


# -------------------------
# Fixed-feature LinearTS baseline
# -------------------------


class LinearTSAgent:
    """
    Linear Thompson Sampling with fixed features:
      phi(x,a) = [x; a]

    This is intentionally simple and will struggle when the ground truth is nonlinear in (x,a).
    """

    def __init__(
        self,
        *,
        ctx_dim: int,
        act_dim: int,
        prior_var: float,
        obs_noise_var: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.ctx_dim = int(ctx_dim)
        self.act_dim = int(act_dim)
        self.phi_dim = self.ctx_dim + self.act_dim
        self.device = device
        self.dtype = dtype

        self.posterior = BayesianLinearRegression(
            dim=self.phi_dim,
            prior_var=float(prior_var),
            obs_noise_var=float(obs_noise_var),
            device=device,
            dtype=dtype,
        )

    def _phi_matrix(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        x: (ctx_dim,)
        actions: (K, act_dim)
        return: (K, phi_dim)
        e.g. suppose x = [0.5, -1.0], and candidate actions (K=2) Tensor([[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5]]).
        then Phi = Tensor(
        [
            [.5,   -1.0,   1.0, 0.0, 2.0 ],
            [ 0.5, -1.0,  -1.0, 3.0, 0.5 ],
        ])
        """
        if x.ndim != 1 or x.shape[0] != self.ctx_dim:
            raise ValueError(f"x must be ({self.ctx_dim},), got {tuple(x.shape)}")
        if actions.ndim != 2 or actions.shape[1] != self.act_dim:
            raise ValueError(f"actions must be (K,{self.act_dim}), got {tuple(actions.shape)}")
        K = int(actions.shape[0])
        x_rep = x.unsqueeze(0).expand(K, -1)
        return torch.cat([x_rep, actions], dim=1)

    def _phi_single(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 1 or a.shape[0] != self.act_dim:
            raise ValueError(f"a must be ({self.act_dim},), got {tuple(a.shape)}")
        return torch.cat([x, a], dim=0)

    def select_action(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        w = self.posterior.sample_weights(n_samples=1, generator=generator)  # (phi_dim,)
        Phi = self._phi_matrix(x, actions)  # (K, phi_dim)
        scores = Phi @ w  # (K,)
        return int(torch.argmax(scores).item())

    def update(self, x: torch.Tensor, a: torch.Tensor, reward: float | torch.Tensor) -> None:
        phi = self._phi_single(x, a)
        self.posterior.update(phi, reward)


# -------------------------
# Two-tower encoder for NeuralLinearTS
# -------------------------


class TwoTowerEncoder(nn.Module):
    """
    Two-tower encoder with fast encode_batch():

      u = f_ctx(x)
      v = f_act(a)
      z = [u; v; u*v], "*" represents elementwise multiplication

    where u,v ∈ R^{tower_dim} so feature_dim = 3*tower_dim. Note that we do not simply use u*v as
    the encoder output; we concatenate u, v and u*v.

    IMPORTANT (encode_batch efficiency):
    - encode_batch(context, actions) computes u once, then computes v for all actions.
    - It does NOT expand the context and re-run f_ctx K times.
    """

    def __init__(self, *, ctx_dim: int, act_dim: int, tower_dim: int) -> None:
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.act_dim = int(act_dim)
        self.tower_dim = int(tower_dim)
        self.feature_dim = 3 * self.tower_dim  # concatenate u, v, u*v

        self.ctx_tower = nn.Sequential(
            nn.Linear(self.ctx_dim, 2 * self.tower_dim),
            nn.ReLU(),
            nn.Linear(2 * self.tower_dim, self.tower_dim),
            nn.Tanh(),
        )
        self.act_tower = nn.Sequential(
            nn.Linear(self.act_dim, 2 * self.tower_dim),
            nn.ReLU(),
            nn.Linear(2 * self.tower_dim, self.tower_dim),
            nn.Tanh(),
        )

    def encode(self, context: torch.Tensor, action: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # context: (ctx_dim,), action: (act_dim,)
        u = self.ctx_tower(context.unsqueeze(0)).squeeze(0)  # (d,)
        v = self.act_tower(action.unsqueeze(0)).squeeze(0)   # (d,)
        return torch.cat([u, v, u * v], dim=0)               # (3d,)

    def encode_batch(self, context: torch.Tensor, actions: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # context: (ctx_dim,), actions: (K, act_dim)
        u = self.ctx_tower(context.unsqueeze(0)).squeeze(0)  # (d,)
        v = self.act_tower(actions)                          # (K, d)
        u_rep = u.unsqueeze(0).expand(v.shape[0], -1)        # (K, d)
        return torch.cat([u_rep, v, u_rep * v], dim=1)       # (K, 3d)


# -------------------------
# Synthetic environment
# -------------------------


@dataclass(frozen=True)
class EnvParams:
    ctx_dim: int
    act_dim: int
    latent_dim: int
    bias: float


class SyntheticTwoTowerEnv:
    """
    Nonlinear ground-truth reward model to have a learnable structure between contexts and actions.

    The ground-truth structure is nonlinear because of tanh. Why tanh? Because it keeps coordinates
    bounded in [-1,1], preventing huge logits and making the reward function stable.

      u* = tanh(Wx x)
      v* = tanh(Wa a)
      logits = <u*, v*> / sqrt(d) + 0.25 * mean((u* * v*)^2) + bias
      mean   = sigmoid(logits)   # mean reward in [0,1]

    IMPORTANT:
    We do NOT sample a Bernoulli( mean ) reward here. Instead we sample a *Gaussian* reward around
    the mean:

      reward = clamp(mean + eps, 0, 1),   eps ~ Normal(0, sigma^2)

    This keeps the example aligned with BayesianLinearRegression (Gaussian observation model) and
    reduces noise so the NeuralLinearTS advantage is easier to see.

    This is designed so a two-tower encoder has a clear advantage over a purely linear phi(x,a) baseline.

    ------------------------------------------------------------------------
    QUICK DEFINITIONS (for example 03; matches the code below)
    ------------------------------------------------------------------------
    - self.p: an EnvParams struct (params for this toy environment).
      It typically contains:
        * latent_dim (a.k.a. d): the hidden embedding size used by the environment, e.g. 16, 32, 64
        * ctx_dim: dimension of raw context vector x, e.g. 32
        * act_dim: dimension of raw action descriptor a, e.g. 32
        * bias: scalar bias added to logits

      Think: x lives in R^{ctx_dim}, each action a_k lives in R^{act_dim},
      and the environment maps both into a shared latent space R^{latent_dim}.

    - latent_dim (d): the dimension of u* and v* after the tanh nonlinearity.
      u* = tanh(Wx x) has shape (d,)
      v* = tanh(Wa a) has shape (d,) for one action, or (K,d) for K actions

    - ctx_dim: dimension of x (the context vector)
    - act_dim: dimension of a (each action feature vector)

    ------------------------------------------------------------------------
    TOY SHAPE EXAMPLES (small numbers)
    ------------------------------------------------------------------------
    Suppose:
      latent_dim = d = 3
      ctx_dim = 4
      act_dim = 2
      K = 5 actions

    Then:
      x            : shape (4,)
      actions      : shape (5,2)  (5 candidate actions, each 2D)
      Wx           : shape (3,4)
      Wa           : shape (3,2)

      Wx @ x       : shape (3,)   -> then tanh -> u*: (3,)
      actions @ Wa.T: (5,2) @ (2,3) = (5,3) -> then tanh -> v*: (5,3)

    ------------------------------------------------------------------------
    TOY NUMERICAL EXAMPLE FOR Wx AND Wa (illustrative only)
    ------------------------------------------------------------------------
    If d=3, ctx_dim=4, act_dim=2, you can imagine:

      Wx = [[ 0.7, -0.2,  0.1,  0.0],
            [ 0.3,  0.5, -0.4,  0.2],
            [-0.1,  0.0,  0.6, -0.3]]   # shape (3,4)

      Wa = [[ 0.2, -0.5],
            [ 0.1,  0.3],
            [-0.4,  0.6]]               # shape (3,2)

    (In the real code we draw these randomly with torch.randn(...) * 0.7)

    ------------------------------------------------------------------------
    WHAT HAPPENS INSIDE mean_reward() (toy walkthrough)
    ------------------------------------------------------------------------
    Using the same toy shapes:
      x: (4,)
      actions: (K=5,2)

    1) u = tanh(Wx @ x)                       -> u: (3,)
       Example:
         Wx @ x = [ 0.9, -0.2, 0.4 ]
         tanh(...) -> u ≈ [0.716, -0.197, 0.379]

    2) v = tanh(actions @ Wa.T)               -> v: (5,3)
       Example:
         actions @ Wa.T might yield a 5x3 matrix, then tanh keeps entries bounded.

    3) base interaction:
         base_k = sum_j v[k,j] * u[j] / sqrt(d)      -> base: (5,)
       Here (v * u) is elementwise multiply, then sum over latent dim.
       The /sqrt(d) keeps logits scale stable as d changes.

    4) extra nonlinear interaction:
         inter_k = mean_j ( (v[k,j] * u[j])^2 )      -> inter: (5,)
       This makes the mean reward depend on more than just a dot product.
       It’s still easy-ish for a two-tower + fusion encoder to learn.

    5) logits and mean reward:
         logits_k = base_k + 0.25 * inter_k + bias
         mean_k   = sigmoid(logits_k)                 -> mean: (5,)
       Each mean_k is the expected reward in [0,1] under this environment.

    ------------------------------------------------------------------------
    WHAT sample_reward() DOES (Gaussian reward)
    ------------------------------------------------------------------------
    - Input: mean tensor with values typically in [0,1], shape (K,) or scalar.
    - Output: same shape, sampled Gaussian reward, then clamped to [0,1]:

        reward = clamp(mean + eps, 0, 1)
        eps ~ Normal(0, sigma^2)

    Toy example:
      mean = tensor([0.10, 0.70, 0.50, 0.95])

    One possible noise draw with sigma=0.10:
      eps  = tensor([+0.03, -0.11, +0.02, +0.06])

    Then:
      mean + eps = tensor([0.13, 0.59, 0.52, 1.01])
      clamp(...) -> tensor([0.13, 0.59, 0.52, 1.00])

    NOTE:
    - Unlike Bernoulli, this produces continuous rewards, reducing variance and making learning curves smoother.
    - If you want to adjust sigma, edit REWARD_NOISE_STD below (intentionally kept in this class).
    """

    # Std of Gaussian observation noise around the mean reward knob: tweak here if you want).
    REWARD_NOISE_STD = 0.10

    def __init__(
        self,
        *,
        params: EnvParams,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        self.p = params
        self.device = device
        self.dtype = dtype
        self.gen = generator

        # self.Wx maps raw contexts x (R^{ctx_dim}) into latent u* (R^{latent_dim}).
        # Shape: (latent_dim, ctx_dim)
        self.Wx = torch.randn(self.p.latent_dim, self.p.ctx_dim, device=device, dtype=dtype, generator=self.gen) * 0.7

        # self.Wa maps raw action vectors a (R^{act_dim}) into latent v* (R^{latent_dim}).
        # Shape: (latent_dim, act_dim)
        #
        # In mean_reward we use Wa.T so that:
        #   actions: (K, act_dim)
        #   Wa.T   : (act_dim, latent_dim)
        #   actions @ Wa.T -> (K, latent_dim)
        self.Wa = torch.randn(self.p.latent_dim, self.p.act_dim, device=device, dtype=dtype, generator=self.gen) * 0.7

    def sample_context(self) -> torch.Tensor:
        # Returns one context vector x in R^{ctx_dim}
        # Shape: (ctx_dim,)
        return torch.randn(self.p.ctx_dim, device=self.device, dtype=self.dtype, generator=self.gen)

    def mean_reward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return the expected mean reward (in [0,1]) for each candidate action.

        x: (ctx_dim,)
        actions: (K, act_dim)
        returns: (K,)
        """
        # u*: (d,)
        u = torch.tanh(self.Wx @ x)

        # v*: (K, d)
        v = torch.tanh(actions @ self.Wa.T)

        # base interaction: (K,)
        # base_k = <v_k, u> / sqrt(d)
        base = (v * u.unsqueeze(0)).sum(dim=1) / (float(self.p.latent_dim) ** 0.5)

        # extra nonlinear interaction: mean((u*v)^2) over latent dim
        # NOTE: This is squared interaction (the intended nonlinear term).
        inter = ((v * u.unsqueeze(0)) ** 2).mean(dim=1)

        logits = base + 0.25 * inter + float(self.p.bias)

        # Mean reward in [0,1]. (Older versions sometimes used Bernoulli(mean);
        # this example uses Gaussian sampling around this mean instead.)
        return torch.sigmoid(logits)

    def sample_reward(self, mean: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        """
        mean: (K,) or scalar, expected reward in [0,1]
        returns: Gaussian sample around mean, clamped to [0,1]

          reward = clamp(mean + eps, 0, 1), eps ~ Normal(0, sigma^2)
        """
        # torch.randn_like(mean, generator=...) is not supported consistently across all builds,
        # so we use torch.randn(mean.shape, ...) explicitly.
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator) * float(self.REWARD_NOISE_STD)
        return (mean + eps).clamp_(0.0, 1.0)


# -------------------------
# Main
# -------------------------


def main() -> None:
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--plot", action="store_true")

    # dimensions + action set
    p.add_argument("--ctx-dim", type=int, default=32)
    p.add_argument("--action-dim", type=int, default=32)
    p.add_argument("--tower-dim", type=int, default=32)  # feature_dim will be 3*tower_dim
    p.add_argument("--n-actions", type=int, default=10_000)
    p.add_argument("--candidate-size", type=int, default=1024, help="<=0 means score all actions each step (can be slow).")

    # warmup and training
    p.add_argument("--warmup-random", type=int, default=500)
    p.add_argument("--train-every", type=int, default=200)
    p.add_argument("--optimizer-steps", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--rebuild-chunk", type=int, default=4096)
    p.add_argument("--action-chunk", type=int, default=256, help="Default chunk size for select_action() scoring.")

    # posterior knobs
    p.add_argument("--prior-var", type=float, default=5.0)
    # Default matches SyntheticTwoTowerEnv.REWARD_NOISE_STD (~0.10 => variance ~0.01) for a clean demo.
    p.add_argument("--obs-noise-var", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)

    # replay
    p.add_argument("--buffer-capacity", type=int, default=100_000)

    # logging
    p.add_argument("--print-every", type=int, default=500)

    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Independent RNG streams
    env_gen = torch.Generator(device=device).manual_seed(args.seed + 1)     # environment randomness
    policy_gen = torch.Generator(device=device).manual_seed(args.seed + 2)  # action selection randomness
    train_gen = torch.Generator(device=device).manual_seed(args.seed + 3)   # replay sampling for SGD
    cand_gen = torch.Generator(device=device).manual_seed(args.seed + 4)    # candidate sampling (used in warmup)

    # ---- Action table (fixed vectors) ----
    n_actions = int(args.n_actions)
    act_dim = int(args.action_dim)
    action_table = torch.randn(n_actions, act_dim, device=device, dtype=dtype, generator=env_gen) * 0.6

    # ---- Environment ----
    ctx_dim = int(args.ctx_dim)
    latent_dim = int(args.tower_dim)
    env = SyntheticTwoTowerEnv(
        params=EnvParams(ctx_dim=ctx_dim, act_dim=act_dim, latent_dim=latent_dim, bias=-0.1),
        device=device,
        dtype=dtype,
        generator=env_gen,
    )

    # ---- Baseline: LinearTS ----
    linear = LinearTSAgent(
        ctx_dim=ctx_dim,
        act_dim=act_dim,
        prior_var=float(args.prior_var),
        obs_noise_var=float(args.obs_noise_var),
        device=device,
        dtype=dtype,
    )

    # ---- NeuralLinearTS: TwoTowerEncoder + MemoryReplayBuffer ----
    encoder = TwoTowerEncoder(ctx_dim=ctx_dim, act_dim=act_dim, tower_dim=latent_dim).to(device=device, dtype=dtype)

    # feature_dim is determined by encoder design (3*tower_dim)
    feature_dim = int(encoder.feature_dim)

    policy_cfg = NeuralLinearTSConfig(
        feature_dim=feature_dim,
        posterior_type="bayes_linear",
        prior_var=float(args.prior_var),
        obs_noise_var=float(args.obs_noise_var),
        lr=float(args.lr),
        action_chunk_size=int(args.action_chunk) if args.action_chunk is not None else None,
        encode_kwargs={},  # default encoder kwargs; you can override per-call
    )

    replay = MemoryReplayBuffer(int(args.buffer_capacity), device=device, dtype=dtype)

    neural = NeuralLinearTS(
        encoder=encoder,
        config=policy_cfg,
        buffer=replay,
        device=device,
        dtype=dtype,
    )

    # ---- Metrics ----
    lin_reward: list[float] = []
    neu_reward: list[float] = []
    lin_regret: list[float] = []  # expected regret (best_mean - chosen_mean on the candidate set)
    neu_regret: list[float] = []

    lin_sel_ms: list[float] = []
    neu_sel_ms: list[float] = []

    # Expected mean diagnostics (on the candidate set)
    best_p_hist: list[float] = []
    p_lin_hist: list[float] = []
    p_neu_hist: list[float] = []

    warmup = int(args.warmup_random)
    horizon = int(args.horizon)

    # --------------------------------------------------------------------
    # Candidate generation via dot-product retrieval (two-tower)
    # --------------------------------------------------------------------
    #
    # A common large-action pattern is:
    #   1) retrieval: get a candidate set using a cheap dot-product u·v
    #   2) reranking: run Thompson Sampling / BLR head on that candidate set
    #
    # Here we simulate retrieval by scoring ALL actions with:
    #   u = ctx_tower(x)    (shape: (tower_dim,))
    #   v_i = act_tower(a_i) for each action a_i in action_table (shape: (tower_dim,))
    #   retrieval_score_i = <u, v_i>
    #
    # Then we take top-M by retrieval_score_i as candidates.
    #
    # IMPORTANT:
    # - act_tower(action_table) depends on encoder params. If act_tower is trained, cached
    #   action embeddings become stale. We refresh the cache after each encoder training phase.
    # - During warmup we keep candidate generation RANDOM to avoid early bias from an untrained
    #   retrieval model (and to ensure broad exploration across the action table).
    #
    # In real applications, retrieval might be a separate model/ANN index. This example focuses
    # on the mechanics: "candidate subset -> select_action(candidates)".
    #
    action_emb_table: Optional[torch.Tensor] = None  # (n_actions, tower_dim)

    def _refresh_action_embeddings() -> None:
        """
        Recompute v_i = act_tower(a_i) for all actions in action_table.

        We do this under no_grad and in eval mode for determinism.
        """
        nonlocal action_emb_table
        was_training = bool(encoder.training)
        encoder.eval()

        with torch.no_grad():
            # Chunked computation to keep memory stable for larger action tables.
            chunk = 8192
            vs: list[torch.Tensor] = []
            for start in range(0, n_actions, chunk):
                A = action_table[start : start + chunk]
                vs.append(encoder.act_tower(A))  # (chunk, tower_dim)
            action_emb_table = torch.cat(vs, dim=0)  # (n_actions, tower_dim)

        if was_training:
            encoder.train()

    # Initial retrieval cache (based on randomly initialized towers)
    _refresh_action_embeddings()

    # Mix some random actions into retrieval candidates to avoid exposure bias.
    # 0.2 means: 80% retrieved (top by score), 20% random.
    RETRIEVAL_RANDOM_FRACTION = 0.2

    def _sample_candidates(x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Return candidate indices (M,) into action_table.

        Candidate generation here mimics a realistic 2-stage system:
          retrieval (cheap) -> rerank (TS on candidates)

        - Warmup: random candidates (broad exploration).
        - After warmup:
            * retrieve top actions by a dot-product score that is ALIGNED with the current BLR head:
                "score(a) ≈ (w_v + w_uv * u)^T v"
                derivation: The two-tower feature is z = [u; v; u*v], where u,v ∈ R^d and u*v is elementwise.
                Partition the posterior weight vector to match w = [w_u; w_v; w_uv] with each in R^d.
                Then score = w^T z = (w_u)^T u + (w_v)^T v + (w_uv)^T u * v. Rewrite the last term, 
                (w_uv)^T u * v = Sum(w_{uv,i} u_i v_i) = (w_uv * u)^T v. Hence
                score = (w_u)^T u + (w_v + w_uv * u)^T v.
                For a fixed context, u is fixed, so (w_u)^T u is a constant w.r.t. action ranking, and action 
                ranking is controlled by (w_v + w_uv * u)^T v.

              where:
                u = ctx_tower(x)
                v = act_tower(a)
                w = posterior mean over z=[u; v; u*v]
            * mix in some random actions for exploration / diversity.
        """
        cs = int(args.candidate_size)
        if cs <= 0 or cs >= n_actions:
            return torch.arange(n_actions, device=device)

        # Warmup: purely random candidate sets (avoid retrieval bias before any learning signal)
        if t <= warmup:
            return torch.randint(0, n_actions, size=(cs,), generator=cand_gen, device=device)

        assert action_emb_table is not None  # (n_actions, tower_dim)

        # How many retrieved vs random
        m_rand = int(round(RETRIEVAL_RANDOM_FRACTION * cs))
        m_rand = max(0, min(m_rand, cs))
        m_top = cs - m_rand

        was_training = bool(encoder.training)
        encoder.eval()

        with torch.no_grad():
            # Context embedding u (tower_dim,)
            u = encoder.ctx_tower(x.unsqueeze(0)).squeeze(0)  # (d,)
            d = int(encoder.tower_dim)

            # Use posterior mean for stable retrieval (not a fresh TS sample).
            mu = neural.posterior.posterior_mean().detach()  # (3d,)
            if mu.numel() != 3 * d:
                raise RuntimeError(f"Expected posterior dim {3*d}, got {mu.numel()}")

            w_u = mu[0:d]
            w_v = mu[d : 2 * d]
            w_uv = mu[2 * d : 3 * d]

            # Retrieval query for dot-product with action embeddings v:
            # score(a) = const + (w_v + w_uv * u)^T v
            q = w_v + (w_uv * u)  # (d,)

            scores = action_emb_table @ q  # (n_actions,)
            top_idx = torch.topk(scores, k=max(1, m_top)).indices  # (m_top,)

        if was_training:
            encoder.train()

        if m_rand == 0:
            return top_idx[:cs]

        # Mix-in random candidates (dedupe + pad if needed)
        rand_idx = torch.randint(0, n_actions, size=(m_rand,), generator=cand_gen, device=device)
        cand = torch.unique(torch.cat([top_idx, rand_idx], dim=0), sorted=False)

        # If dedupe made it too short, pad with more random indices
        while cand.numel() < cs:
            extra = torch.randint(0, n_actions, size=(cs - cand.numel(),), generator=cand_gen, device=device)
            cand = torch.unique(torch.cat([cand, extra], dim=0), sorted=False)

        return cand[:cs]

    # ---- Loop ----
    for t in range(1, horizon + 1):
        x = env.sample_context()  # (ctx_dim,)

        cand_idx = _sample_candidates(x, t)                   # (M,)
        cand_actions = action_table.index_select(0, cand_idx)  # (M, act_dim)

        # Oracle best expected mean among *this* candidate set (cheap + consistent)
        with torch.no_grad():
            probs = env.mean_reward(x, cand_actions)  # (M,) expected mean reward in [0,1]
            best_p = float(probs.max().item())

        # ----- LinearTS -----
        t0 = time.perf_counter()
        if t <= warmup:
            j_lin = int(torch.randint(0, cand_actions.shape[0], (1,), generator=policy_gen, device=device).item())
        else:
            j_lin = linear.select_action(x, cand_actions, generator=policy_gen)
        lin_sel_ms.append(1000.0 * (time.perf_counter() - t0))

        a_lin = cand_actions[j_lin]
        p_lin = float(probs[j_lin].item())

        # Gaussian reward observation around mean (demo-friendly); returns scalar tensor -> float.
        r_lin = float(env.sample_reward(torch.tensor(p_lin, device=device, dtype=dtype), generator=env_gen).item())

        linear.update(x, a_lin, r_lin)
        lin_reward.append(r_lin)
        lin_regret.append(best_p - p_lin)

        # ----- NeuralLinearTS -----
        t0 = time.perf_counter()
        if t <= warmup:
            j_neu = int(torch.randint(0, cand_actions.shape[0], (1,), generator=policy_gen, device=device).item())
        else:
            # NeuralLinearTS will use encoder.encode_batch() internally; config.action_chunk_size controls chunking.
            j_neu = neural.select_action(x, cand_actions, generator=policy_gen)
        neu_sel_ms.append(1000.0 * (time.perf_counter() - t0))

        a_neu = cand_actions[j_neu]
        p_neu = float(probs[j_neu].item())

        # Gaussian reward observation around mean (demo-friendly); returns scalar tensor -> float.
        r_neu = float(env.sample_reward(torch.tensor(p_neu, device=device, dtype=dtype), generator=env_gen).item())

        best_p_hist.append(best_p)
        p_lin_hist.append(p_lin)
        p_neu_hist.append(p_neu)

        neural.update(x, a_neu, r_neu)
        neu_reward.append(r_neu)
        neu_regret.append(best_p - p_neu)

        # periodic encoder training + posterior rebuild
        if args.train_every > 0 and (t % int(args.train_every) == 0):
            if len(neural.buffer) >= max(8, int(args.batch_size)):
                _ = neural.train_encoder(
                    optimizer_steps=int(args.optimizer_steps),
                    batch_size=int(args.batch_size),
                    generator=train_gen,
                )

                # If act_tower was updated by training, refresh retrieval cache so dot-product candidates track the latest encoder.
                _refresh_action_embeddings()

                neural.rebuild_posterior(chunk_size=int(args.rebuild_chunk))

        # progress
        if t in {50, 100, 200} or (t % int(args.print_every) == 0) or (t == horizon):
            lin_cum_reg = float(sum(lin_regret))
            neu_cum_reg = float(sum(neu_regret))
            lin_ma = moving_avg(lin_reward, 200)
            neu_ma = moving_avg(neu_reward, 200)

            lin_ms = moving_avg(lin_sel_ms, 200)
            neu_ms = moving_avg(neu_sel_ms, 200)

            ma_best_p = moving_avg(best_p_hist, 200)
            ma_p_lin = moving_avg(p_lin_hist, 200)
            ma_p_neu = moving_avg(p_neu_hist, 200)

            print(
                f"[t={t:5d}] "
                f"LinearTS  cum_reg={lin_cum_reg:10.2f}  ma_reward={lin_ma:+.3f}  sel_ms={lin_ms:6.2f} | "
                f"NeuralTS  cum_reg={neu_cum_reg:10.2f}  ma_reward={neu_ma:+.3f}  sel_ms={neu_ms:6.2f} | "
                f"ma_best_p={ma_best_p:.3f}  ma_p_lin={ma_p_lin:.3f}  ma_p_neu={ma_p_neu:.3f} | "
            )

    # ---- Summary ----
    print("\nFinal:")
    print(f"  LinearTS cumulative expected regret: {sum(lin_regret):.2f}")
    print(f"  NeuralTS cumulative expected regret: {sum(neu_regret):.2f}")
    print(f"  LinearTS moving avg reward (last 200): {moving_avg(lin_reward, 200):.3f}")
    print(f"  NeuralTS moving avg reward (last 200): {moving_avg(neu_reward, 200):.3f}")
    print(f"  LinearTS avg select_action ms (last 200): {moving_avg(lin_sel_ms, 200):.2f}")
    print(f"  NeuralTS avg select_action ms (last 200): {moving_avg(neu_sel_ms, 200):.2f}")

    if args.plot:
        lin_cum: list[float] = []
        neu_cum: list[float] = []
        s1 = 0.0
        s2 = 0.0
        for r1, r2 in zip(lin_regret, neu_regret):
            s1 += r1
            s2 += r2
            lin_cum.append(s1)
            neu_cum.append(s2)

        maybe_plot({"LinearTS": lin_cum, "NeuralTS": neu_cum}, title="Cumulative expected regret (candidate-oracle)")
        maybe_plot({"LinearTS": lin_reward, "NeuralTS": neu_reward}, title="Reward (Gaussian sample) per step")
        maybe_plot({"LinearTS": lin_sel_ms, "NeuralTS": neu_sel_ms}, title="select_action latency (ms)")


if __name__ == "__main__":
    main()
