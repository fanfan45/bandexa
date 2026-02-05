# examples/02_mnist_bandit.py
"""
MNIST contextual bandit (bandit classification) example.

What this shows
---------------
A fully online contextual bandit loop using *real data*:

  select action  -> observe reward -> update posterior -> (periodic) train encoder -> rebuild posterior

We compare two agents:

1) LinearTS (fixed features):
   - Uses a Bayesian linear posterior directly on a fixed feature map.
   - Here we use a linear-classifier-style feature map:
       phi(x, a_k) = [a_k ; x_flat ⊗ a_k]
     where a_k is a one-hot label action (k in {0..9}).
     This gives each class its own linear weights on pixels (plus a bias term).

   IMPORTANT PRACTICAL NOTE (to keep the baseline feasible):
   --------------------------------------------------------
   When using a ResNet encoder, the neural agent consumes a *preprocessed* image context
   shaped (3, 224, 224). If we fed that same (3,224,224) tensor to LinearTSAgent, the
   fixed-feature dimension would become enormous:

     D = 3*224*224 = 150,528
     phi_dim = K + K*D  (K=10)  -> ~1,505,290 parameters

   A Bayesian linear regression posterior that stores dense matrices at that dimension
   is intractable in memory.

   Therefore, LinearTSAgent always runs on the *raw* MNIST image tensor (1,28,28),
   regardless of which neural backbone is chosen. This way, the baseline remains:
     D = 1*28*28 = 784
     phi_dim = 10 + 10*784 = 7,850
   which is a reasonable “linear classifier” baseline for MNIST.

2) NeuralLinearTS (learned features):
   - Uses NeuralLinearTS with a trainable encoder z = encoder(image, action).
   - Default encoder is a small ConvNet.
   - Optional: use a pretrained ResNet backbone (downloads weights by default),
     via examples/_resnet_backbone.py.
     The backbone manages its own trainability; by default it freezes most layers and
     unfreezes only the last residual stage (layer4).
   - The bandit posterior is built internally by NeuralLinearTS (here: bayes_linear).

Bandit feedback
---------------
At each step we sample a MNIST image with true label y.
Actions are labels {0..9}. Reward is:
  r = 1 if chosen_label == y else 0

We only observe the reward for the chosen action (bandit-style),
even though in MNIST you do have access to the full label.

Regret
------
The best achievable reward per step is 1 (choose the correct label), so:
  instant_regret_t = 1 - r_t
  cumulative_regret = (# mistakes)

Run
---
  python examples/02_mnist_bandit.py
  python examples/02_mnist_bandit.py --help
  python examples/02_mnist_bandit.py --device cuda
  python examples/02_mnist_bandit.py --encoder resnet18 --device cuda

Notes
-----
- Requires torchvision.
- Data downloads to --data-dir (default: examples/data).

Replay buffer note
------------------
NeuralLinearTS logs (context, action, reward) to a replay buffer.

This example builds the replay buffer in the script and passes it into NeuralLinearTS,
so the policy remains buffer-backend-agnostic.

- For convnet contexts (1,28,28), an in-memory buffer is usually fine.
- For ResNet contexts (3,224,224), storing many samples in RAM can be heavy,
  so this example supports a disk-backed buffer.

If your in-memory buffer does not implement iter_batches(), this script wraps it with a
small adapter that provides iter_batches() by batching over buffer.all().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn

from bandexa.posterior.bayes_linear import BayesianLinearRegression
from bandexa.policies.neural_linear_ts import NeuralLinearTS, NeuralLinearTSConfig


# -------------------------
# Utilities
# -------------------------


def moving_avg(xs: list[float], window: int) -> float:
    if not xs:
        return 0.0
    w = min(window, len(xs))
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


def maybe_flush(agent: object) -> None:
    """
    Flush buffered experience to durable storage, if supported.

    - Disk buffers commonly implement flush() or policy wrappers implement flush_buffer().
    - In-memory buffers typically do nothing.
    """
    if hasattr(agent, "flush_buffer"):
        agent.flush_buffer()  # type: ignore[call-arg]
        return
    buf = getattr(agent, "buffer", None)
    if buf is not None and hasattr(buf, "flush"):
        buf.flush()  # type: ignore[call-arg]


# -------------------------
# Replay buffer adapter (for older in-memory buffers)
# -------------------------


@dataclass(frozen=True)
class _SimpleBatch:
    """
    Minimal batch container compatible with ReplayBatch protocol in bandexa.buffers.base.

    We intentionally keep this local to the example to avoid coupling to internal buffer types.
    """
    contexts: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


def build_replay_buffer(
    *,
    kind: str,
    capacity: int,
    device: torch.device,
    dtype: torch.dtype,
    root_dir: str,
    disk_shard_size: int,
    disk_cache_shards: int,
) -> object:
    """
    Build a replay buffer instance for the example and return it.

    We keep the construction logic here (in the example) so NeuralLinearTS stays
    agnostic to buffer backend.

    - memory: MemoryReplayBuffer(capacity, device=..., dtype=...)
    - disk:   DiskReplayBuffer(root_dir=..., shard_size=..., cache_shards=..., device=..., dtype=..., [capacity=?])

    For disk buffers, constructor signatures vary; we try a couple of common patterns.
    """
    if kind == "memory":
        from bandexa.buffers.memory_replay import MemoryReplayBuffer  # expected module/class name

        try:
            buf = MemoryReplayBuffer(int(capacity), device=device, dtype=dtype)
        except TypeError:
            # fallback: older signature may be (capacity, device, dtype) positionally
            buf = MemoryReplayBuffer(int(capacity), device, dtype)  # type: ignore[misc]
        return buf

    if kind == "disk":
        from bandexa.buffers.disk_replay import DiskReplayBuffer  # expected module/class name

        # Try signatures from most-specific to least-specific.
        # 1) root_dir, capacity, shard_size, cache_shards, device, dtype
        try:
            return DiskReplayBuffer(
                root_dir=root_dir,
                capacity=int(capacity),
                shard_size=int(disk_shard_size),
                cache_shards=int(disk_cache_shards),
                device=device,
                dtype=dtype,
            )
        except TypeError:
            pass

        # 2) root_dir, shard_size, cache_shards, device, dtype  (capacity handled internally)
        try:
            return DiskReplayBuffer(
                root_dir=root_dir,
                shard_size=int(disk_shard_size),
                cache_shards=int(disk_cache_shards),
                device=device,
                dtype=dtype,
            )
        except TypeError:
            pass

        # 3) root_dir, shard_size, device, dtype, cache_shards  (alternate kw order)
        try:
            return DiskReplayBuffer(
                root_dir=root_dir,
                shard_size=int(disk_shard_size),
                device=device,
                dtype=dtype,
                cache_shards=int(disk_cache_shards),
            )
        except TypeError as e:
            raise TypeError(
                "DiskReplayBuffer constructor signature did not match any supported patterns for this example. "
                "Update build_replay_buffer() to match your DiskReplayBuffer API."
            ) from e

    raise ValueError(f"Unknown buffer kind: {kind!r}")


# -------------------------
# LinearTS baseline
# -------------------------


class LinearTSAgent:
    """
    Linear Thompson Sampling baseline with a fixed feature map suitable for MNIST classification.

    Actions are one-hot label vectors a_k in R^K, K=10.
    Context is the image x in R^(1,28,28).

    Feature map:
      phi(x, a_k) = concat( a_k, kron(a_k, x_flat) )

    Interpretation:
      - kron(a_k, x_flat) is a block vector with K blocks of length D=784
      - because a_k is one-hot, exactly one block equals x_flat and others are zeros
      - the posterior learns a separate linear weight vector per class (plus bias via a_k)

        Small example: number of actions/classes: K = 3 (labels 0,1,2)
        image flattened dimension: D = 4 (instead of 784)
        kron part for action a_0 = [1, 0, 0] is [10,20,30,40, 0,0,0,0, 0,0,0,0], so
        phi(x, a_0) = [1,0,0, 10,20,30,40, 0,0,0,0, 0,0,0,0], likewise
        phi(x, a_1) = [0,1,0, 0,0,0,0, 10,20,30,40, 0,0,0,0], that's exactly what _phi_single() constructs.

        "_phi_matrix(x, actions)" builds Phi for all K candidate actions at once, e.g.,
        PHI(x) = [phi(x, a_0),
                  phi(x, a_1),
                  phi(x, a_2),
                      ...   ]
        for the example above:
        PHI(x) = [
                    [1,0,0, 10,20,30,40, 0,0,0,0, 0,0,0,0],
                    [0,1,0, 0,0,0,0, 10,20,30,40, 0,0,0,0],
                    [0,0,1, 0,0,0,0, 0,0,0,0, 10,20,30,40],
                  ]

        Why this makes LinearTS a “proper” linear classifier baseline:
        The posterior maintains weights w ∈ R^(K + K*D), e.g. R^(3 + 3*4) = R^15 in this toy case:
          - bias weights per class b ∈ R^K (from the a_k block)
          - per-class pixel weights W ∈ R^(K*D) (from the kron block)
        Score for action k becomes: score(k) = b_k + (W_k)^T x
        That is multinomial linear classification (one-vs-rest style scoring),
        trained online via the Bayesian posterior.
    """

    def __init__(
        self,
        *,
        n_actions: int,
        image_shape: tuple[int, int, int],
        prior_var: float,
        obs_noise_var: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        C, H, W = image_shape
        if C * H * W <= 0:
            raise ValueError("invalid image_shape")

        self.K = int(n_actions)
        self.D = int(C * H * W)
        self.phi_dim = self.K + self.K * self.D

        self.device = device
        self.dtype = dtype

        # Bayesian linear posterior over the fixed feature map phi(x,a)
        self.posterior = BayesianLinearRegression(
            dim=self.phi_dim,
            prior_var=float(prior_var),
            obs_noise_var=float(obs_noise_var),
            device=device,
            dtype=dtype,
        )

    def _phi_matrix(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Build Phi for all candidate one-hot actions.
        x: (C,H,W) on self.device
        actions: (K,K) one-hot
        returns Phi: (K, phi_dim)
        """
        if actions.ndim != 2 or actions.shape[0] != self.K or actions.shape[1] != self.K:
            raise ValueError(f"actions must be (K,K) one-hot, got {tuple(actions.shape)}")

        x_flat = x.reshape(-1)  # (D,)
        # Build kron(a_k, x_flat) efficiently for one-hot actions: put x_flat in block k.
        blocks = torch.zeros((self.K, self.K * self.D), device=self.device, dtype=self.dtype)
        for k in range(self.K):
            blocks[k, k * self.D : (k + 1) * self.D] = x_flat

        return torch.cat([actions, blocks], dim=1)  # (K, K + K*D)

    def _phi_single(self, x: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        """Build phi(x,a) for one chosen one-hot action."""
        if a_onehot.shape != (self.K,):
            raise ValueError(f"a_onehot must have shape ({self.K},), got {tuple(a_onehot.shape)}")
        x_flat = x.reshape(-1)  # (D,)
        blocks = torch.zeros((self.K * self.D,), device=self.device, dtype=self.dtype)
        k = int(torch.argmax(a_onehot).item())
        blocks[k * self.D : (k + 1) * self.D] = x_flat
        return torch.cat([a_onehot, blocks], dim=0)  # (phi_dim,)

    def select_action(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """
        1) sample one weight vector w from posterior
        2) compute scores = Phi @ w giving K scores
        3) choose argmax

        That's why _phi_matrix() matters: it lets us score all candidate labels in one matmul.
        """
        w = self.posterior.sample_weights(n_samples=1, generator=generator)  # (phi_dim,)
        Phi = self._phi_matrix(x, actions)  # (K, phi_dim)
        scores = Phi @ w  # (K,)
        return int(torch.argmax(scores).item())

    def update(self, x: torch.Tensor, a_onehot: torch.Tensor, reward: float | torch.Tensor) -> None:
        phi = self._phi_single(x, a_onehot)
        self.posterior.update(phi, reward)


# -------------------------
# Learned encoder for NeuralLinearTS
# -------------------------


class SmallConvBackbone(nn.Module):
    """
    Small ConvNet backbone for MNIST (fast).

    Input:  (B, 1, 28, 28)
    Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class ImageActionEncoder(nn.Module):
    """
    Joint encoder z = phi(image, action) for NeuralLinearTS.

    NeuralLinearTS expects an encoder to provide:
      - encode(context, action, **kwargs) -> z of shape (feature_dim,)
      - optionally encode_batch(context, actions, **kwargs) -> Z of shape (n_actions, feature_dim)

    In this MNIST example, the *context* is an image tensor and the *action* is a one-hot
    label vector. When you build an encoder from scratch you can model phi([context; action])
    directly with a single network over the concatenated input.

    Here we use a more parameter-efficient design that works well with both:
      - a small trainable ConvNet backbone, and
      - a pretrained vision backbone (e.g., ResNet) where only some layers are trainable.

    Architecture (two-path / “two-tower”-like):
      1) backbone(image) -> h_img in R^{backbone_out_dim}
      2) action_dense(action_onehot) -> h_act in R^{hidden}
      3) fuse([h_img; h_act]) -> z in R^{feature_dim}

    This lets us reuse a fixed image representation while still learning how actions interact
    with that representation via a small number of trainable parameters (action_dense + fuse).
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        backbone_out_dim: int,
        n_actions: int = 10,
        feature_dim: int = 64,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)

        self.backbone = backbone
        self.action_dense = nn.Sequential(
            nn.Linear(self.n_actions, hidden),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(backbone_out_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def encode(self, context: torch.Tensor, action: torch.Tensor, **kwargs: object) -> torch.Tensor:
        img = context.unsqueeze(0)  # (1,C,H,W)
        a = action.unsqueeze(0)  # (1,K)

        img_feat = self.backbone(img)  # (1, Bdim)
        a_feat = self.action_dense(a)  # (1, H)
        z = self.fuse(torch.cat([img_feat, a_feat], dim=1))  # (1, d)
        return z.squeeze(0)  # (d,)

    def encode_batch(self, context: torch.Tensor, actions: torch.Tensor, **kwargs: object) -> torch.Tensor:
        K = int(actions.shape[0])
        img = context.unsqueeze(0).expand(K, *context.shape)  # (K,C,H,W)

        img_feat = self.backbone(img)  # (K,Bdim)
        a_feat = self.action_dense(actions)  # (K,H)
        z = self.fuse(torch.cat([img_feat, a_feat], dim=1))  # (K,d)
        return z


# -------------------------
# Main
# -------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="examples/data")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--plot", action="store_true")

    # Posterior knobs (modeling params)
    p.add_argument("--posterior-prior-var", type=float, default=5.0)
    p.add_argument(
        "--posterior-obs-noise-var",
        type=float,
        default=1.0,
        help="Observation noise variance for the Bayesian linear posterior (reward is 0/1 here).",
    )

    # NeuralLinearTS knobs
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--train-every", type=int, default=200)
    p.add_argument("--optimizer-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rebuild-chunk", type=int, default=2048)
    p.add_argument("--action-chunk", type=int, default=None)

    # Replay buffer backend: for ResNet contexts, disk is often the right default.
    p.add_argument(
        "--buffer-backend",
        type=str,
        default="auto",
        choices=["auto", "memory", "disk"],
        help=(
            "Replay buffer backend for NeuralLinearTS. "
            "'auto' uses 'disk' for ResNet encoders and 'memory' for convnet."
        ),
    )
    p.add_argument(
        "--buffer-dir",
        type=str,
        default="examples/data/mnist_replay",
        help="Directory for disk-backed replay buffer (used when --buffer-backend=disk or auto->disk).",
    )
    p.add_argument(
        "--buffer-capacity",
        type=int,
        default=50_000,
        help="Replay capacity (memory: max stored; disk: logical capacity if supported).",
    )
    p.add_argument(
        "--disk-shard-size",
        type=int,
        default=2048,
        help="Disk replay shard size (number of samples per shard).",
    )
    p.add_argument(
        "--disk-cache-shards",
        type=int,
        default=2,
        help="How many shards to keep cached in memory for sampling (disk replay).",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help=(
            "If using a disk buffer, call flush() every N steps (0 disables). "
            "Flushing writes the in-progress shard to disk (safer for long runs)."
        ),
    )

    # Encoder choice
    p.add_argument(
        "--encoder",
        type=str,
        default="convnet",
        choices=["convnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help=(
            "Encoder backbone. ResNet uses pretrained weights by default. "
            "Trainability is managed by the backbone itself (see examples/_resnet_backbone.py)."
        ),
    )
    
    p.add_argument(
        "--save-path",
        type=str,
        default="examples/data/neuralts_mnist_inference.pt",
        help="Where to save an inference-only checkpoint at the end of the run.",
    )
    p.add_argument(
        "--sanity-check-save-load",
        action="store_true",
        help="If set, saves an inference checkpoint at t==horizon, reloads it into a fresh agent, "
        "and prints probe deltas to verify save/load correctness.",
    )
    p.add_argument("--warmup-random", type=int, default=1000, help="First N steps choose actions uniformly at random.")
    # Note: Warmup curves are identical for different encoders when random generator is used for reproducibility.
    # During warmup (t <= 1000), both runs, Linear and Ecnoder (convnet/resnet), pick actions uniformly at random
    # (same policy_gen seed, same device → same random action sequence). The dataset stream is also reproducible
    # (same dl_gen seed → same image/label order). Therefore, the realized rewards/regrets are identical, regardless
    # of encoder type, buffer type, or posterior contents.

    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Decide buffer backend (example-level; we build the buffer here and pass it into NeuralLinearTS)
    if args.buffer_backend == "auto":
        buffer_backend = "disk" if args.encoder != "convnet" else "memory"
    else:
        buffer_backend = args.buffer_backend
    print("buffer Type:", buffer_backend)

    # Reproducibility (global RNG); DataLoader gets its own CPU generator
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Independent RNG streams
    policy_gen = torch.Generator(device=device).manual_seed(args.seed + 1)  # sampling / action selection
    train_gen = torch.Generator(device=device).manual_seed(args.seed + 2)  # replay sampling during encoder training
    dl_gen = torch.Generator(device="cpu").manual_seed(args.seed + 3)  # DataLoader MUST be CPU

    # ---- Dataset ----
    try:
        from torchvision import datasets, transforms  # type: ignore
    except Exception as e:
        raise RuntimeError("This example requires torchvision. Install it (e.g., pip install torchvision).") from e

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Base dataset transform:
    # We ALWAYS load MNIST as a tensor in (1,28,28) in [0,1].
    #
    # This is important because:
    #   - LinearTS baseline should always use the raw (1,28,28) tensor (see docstring note above).
    #   - For ResNet, we can further transform this base tensor into a (3,224,224) ImageNet-normalized input.
    base_tfm = transforms.ToTensor()  # (1,28,28), converts 0 - 255 pixel values to float32 in [0,1]

    # Additional transforms for the neural encoder if we pick a ResNet backbone.
    #
    # ResNet expects 3-channel inputs; we upsample MNIST and normalize with ImageNet stats.
    #
    # We now keep the dataset in raw 28x28 form and apply the ResNet preprocessing ONLY for the neural agent.
    if args.encoder == "convnet":
        # Neural agent consumes the same (1,28,28) tensor as the dataset output.
        tfm_neural = None
        image_shape_neural = (1, 28, 28)
    else:
        # transforms.Lambda(...): duplicates the single grayscale channel into 3 channels.
        # transforms.Resize((224,224)) + Normalize(...): standard ImageNet preprocessing for ResNet.
        tfm_neural = transforms.Compose(
            [
                transforms.Lambda(lambda t: t.expand(3, -1, -1)),  # (1,28,28) -> (3,28,28)
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image_shape_neural = (3, 224, 224)

    # Linear baseline always uses raw MNIST (1,28,28).
    image_shape_linear = (1, 28, 28)

    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=base_tfm)

    # Shuffle for a more realistic online stream; generator makes it reproducible
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=dl_gen,
    )

    def infinite_stream(dl):
        """
        This re-iterates the DataLoader each epoch.
        When it finishes, the while True: loop starts another pass, so it re-iterates from the beginning again.
        if the DataLoader has shuffle=True, each new pass will typically produce a new shuffled order
        (and if you pass a fixed generator, that order is reproducible across runs).
        """
        while True:
            for batch in dl:
                yield batch

    stream = infinite_stream(loader)

    # ---- Actions (labels 0..9 as one-hot vectors) ----
    K = 10
    actions = torch.eye(K, device=device, dtype=dtype)  # (10,10)

    chosen_lin = torch.zeros(K, dtype=torch.long)
    chosen_neu = torch.zeros(K, dtype=torch.long)

    # ---- Agents ----
    linear = LinearTSAgent(
        n_actions=K,
        image_shape=image_shape_linear,
        prior_var=float(args.posterior_prior_var),
        obs_noise_var=float(args.posterior_obs_noise_var),
        device=device,
        dtype=dtype,
    )

    # Build backbone + encoder
    if args.encoder == "convnet":
        backbone_out = 64
        backbone = SmallConvBackbone(out_dim=backbone_out)
    else:
        from _resnet_backbone import ResNetBackboneConfig, build_resnet_backbone  # type: ignore

        cfg = ResNetBackboneConfig(
            arch=args.encoder,
            device=device,
            dtype=dtype,
        )
        backbone = build_resnet_backbone(cfg)

        # Infer output dim with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros((1, *image_shape_neural), device=device, dtype=dtype)
            backbone_out = int(backbone(dummy).shape[1])

    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone params: trainable={n_trainable} / total={n_total}")

    encoder = ImageActionEncoder(
        backbone=backbone,
        backbone_out_dim=backbone_out,
        n_actions=K,
        feature_dim=int(args.feature_dim),
        hidden=128,
    )

    # Build replay buffer (in the example) and pass it into NeuralLinearTS
    replay_buffer = build_replay_buffer(
        kind=str(buffer_backend),
        capacity=int(args.buffer_capacity),
        device=device,
        dtype=dtype,
        root_dir=str(args.buffer_dir),
        disk_shard_size=int(args.disk_shard_size),
        disk_cache_shards=int(args.disk_cache_shards),
    )

    # Build policy config (NeuralLinearTS reads its knobs from config)
    policy_cfg = NeuralLinearTSConfig(
        feature_dim=int(args.feature_dim),
        posterior_type="bayes_linear",
        prior_var=float(args.posterior_prior_var),
        obs_noise_var=float(args.posterior_obs_noise_var),
        lr=1e-3,
    )

    neural = NeuralLinearTS(
        encoder=encoder,
        config=policy_cfg,
        buffer=replay_buffer,
        device=device,
        dtype=dtype,
    )

    # ---- Metrics ----
    lin_rewards: list[float] = []
    neu_rewards: list[float] = []
    lin_regrets: list[float] = []
    neu_regrets: list[float] = []

    def _probe_numbers(agent: NeuralLinearTS, img_chw: torch.Tensor, actions_ka: torch.Tensor, a_idx: int) -> tuple[float, float]:
        """
        Deterministic probe scalar(s) to sanity-check save/load:

        - score: z(img,a)^T mu  (uses encoder + posterior mean)
        - mu_norm: ||mu||_2     (a scalar summary of posterior mean)
        """
        if isinstance(agent.encoder, nn.Module):
            agent.encoder.eval()

        with torch.no_grad():
            z = agent.encoder.encode(img_chw, actions_ka[a_idx])  # (d,)
            mu = agent.posterior.posterior_mean().detach()        # (d,)
            score = float((z @ mu).detach().cpu().item())
            mu_norm = float(mu.norm(p=2).cpu().item())
        return score, mu_norm

    # ---- Online loop ----
    horizon = int(args.horizon)
    warmup = int(args.warmup_random)

    for t in range(1, horizon + 1):
        (img_base, y) = next(stream)  # img_base: (1,1,28,28) because batch_size=1 and transform=ToTensor()
        img_base = img_base.squeeze(0)  # (1,28,28) on CPU
        y_int = int(y.item())

        # LinearTS ALWAYS uses the raw MNIST tensor:
        img_lin = img_base.to(device=device, dtype=dtype)  # (1,28,28)

        # Neural agent uses either:
        #   - the same raw tensor for convnet
        #   - ResNet-preprocessed tensor for resnet backbones
        if args.encoder == "convnet":
            img_neu = img_lin
        else:
            if tfm_neural is None:
                raise RuntimeError("Internal error: tfm_neural is None for a ResNet encoder.")
            img_neu_cpu = tfm_neural(img_base)  # (3,224,224) on CPU
            img_neu = img_neu_cpu.to(device=device, dtype=dtype)

        best = 1.0  # oracle best expected reward for classification

        # --- LinearTS ---
        if t <= warmup:
            a_idx_lin = int(torch.randint(0, K, (1,), generator=policy_gen, device=device).item())
        else:
            a_idx_lin = linear.select_action(img_lin, actions, generator=policy_gen)
        chosen_lin[a_idx_lin] += 1

        r_lin = 1.0 if a_idx_lin == y_int else 0.0
        linear.update(img_lin, actions[a_idx_lin], r_lin)
        lin_rewards.append(float(r_lin))
        lin_regrets.append(float(best - r_lin))

        # --- NeuralLinearTS ---
        if t <= warmup:
            a_idx_neu = int(torch.randint(0, K, (1,), generator=policy_gen, device=device).item())
        else:
            a_idx_neu = neural.select_action(img_neu, actions, generator=policy_gen)
        chosen_neu[a_idx_neu] += 1

        r_neu = 1.0 if a_idx_neu == y_int else 0.0
        neural.update(img_neu, actions[a_idx_neu], r_neu)
        neu_rewards.append(float(r_neu))
        neu_regrets.append(float(best - r_neu))

        # Optional: flush disk buffer periodically so the in-progress shard is written.
        # This is most relevant for long runs or if you want crash-safe accumulation.
        if buffer_backend == "disk" and int(args.flush_every) > 0 and (t % int(args.flush_every) == 0):
            maybe_flush(neural)

        # periodic encoder training + posterior rebuild
        if args.train_every > 0 and (t % int(args.train_every) == 0):
            # If using disk buffer, flushing here ensures that any sampling logic that prefers
            # reading from completed shards sees more of the latest data.
            if buffer_backend == "disk":
                maybe_flush(neural)

            if len(neural.buffer) >= max(8, int(args.batch_size)):
                _ = neural.train_encoder(
                    optimizer_steps=int(args.optimizer_steps),
                    batch_size=int(args.batch_size),
                    generator=train_gen,
                )
                neural.rebuild_posterior(chunk_size=int(args.rebuild_chunk))

        # progress
        if t in {50, 100, 200} or (t % 500 == 0) or (t == horizon):
            lin_cum_reg = float(sum(lin_regrets))
            neu_cum_reg = float(sum(neu_regrets))
            lin_acc = moving_avg(lin_rewards, window=200)
            neu_acc = moving_avg(neu_rewards, window=200)

            print(
                f"[t={t:5d}] "
                f"LinearTS  cum_reg={lin_cum_reg:8.1f}  ma_acc={lin_acc:.3f} | "
                f"NeuralTS  cum_reg={neu_cum_reg:8.1f}  ma_acc={neu_acc:.3f} | "
                f"device={device.type} encoder={args.encoder} buffer={buffer_backend} | "
                # f"chosen_lin={chosen_lin.tolist()} | " # uncomment for debugging
                # f"chosen_neu={chosen_neu.tolist()}".   # uncomment for debugging
            )

        # --- End-of-run checkpoint + sanity check for save/load ---
        if t == horizon and args.sanity_check_save_load:
            # Make sure disk buffer has written its last partial shard before exiting.
            if buffer_backend == "disk":
                maybe_flush(neural)

            ckpt_path = str(Path(args.save_path))
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

            probe_a_idx = int(a_idx_neu)

            score_before, mu_norm_before = _probe_numbers(neural, img_neu, actions, probe_a_idx)

            # Save inference-only checkpoint
            neural.save_inference(ckpt_path)

            # Recreate a fresh agent with the SAME architecture/config, then load
            if args.encoder == "convnet":
                backbone_out2 = 64
                backbone2 = SmallConvBackbone(out_dim=backbone_out2)
            else:
                from _resnet_backbone import ResNetBackboneConfig, build_resnet_backbone  # type: ignore
                cfg2 = ResNetBackboneConfig(
                    arch=args.encoder,
                    device=device,
                    dtype=dtype,
                )
                backbone2 = build_resnet_backbone(cfg2)
                with torch.no_grad():
                    dummy2 = torch.zeros((1, *image_shape_neural), device=device, dtype=dtype)
                    backbone_out2 = int(backbone2(dummy2).shape[1])

            encoder2 = ImageActionEncoder(
                backbone=backbone2,
                backbone_out_dim=backbone_out2,
                n_actions=K,
                feature_dim=int(args.feature_dim),
                hidden=128,
            )

            # For the sanity-check agent, we keep a small in-memory buffer, since we don't load buffer anyway.
            # We still wrap it if it doesn't implement iter_batches().
            replay_buffer2 = build_replay_buffer(
                kind="memory",
                capacity=1,
                device=device,
                dtype=dtype,
                root_dir=str(args.buffer_dir),
                disk_shard_size=int(args.disk_shard_size),
                disk_cache_shards=int(args.disk_cache_shards),
            )

            neural2 = NeuralLinearTS(
                encoder=encoder2,
                config=policy_cfg,
                buffer=replay_buffer2,
                device=device,
                dtype=dtype,
            )

            neural2.load_inference(ckpt_path, map_location="cpu")

            score_after, mu_norm_after = _probe_numbers(neural2, img_neu, actions, probe_a_idx)

            # show deltas (these should be ~0 if load was correct)
            print("\n[Sanity check: save/load inference]")
            print(f"  checkpoint: {ckpt_path}")
            print(f"  probe action idx: {probe_a_idx}")
            print(
                f"  score before: {score_before:+.8f}   after: {score_after:+.8f}   "
                f"delta: {score_after - score_before:+.3e}"
            )
            print(
                f"  ||mu||2 before: {mu_norm_before:+.8f}   after: {mu_norm_after:+.8f}   "
                f"delta: {mu_norm_after - mu_norm_before:+.3e}"
            )

            # This is just an object-identity check: fresh agent should not share the same tensor storage.
            same_storage = (
                neural.posterior.precision_matrix().data_ptr()
                == neural2.posterior.precision_matrix().data_ptr()
            )
            print(f"  same precision storage as original? {same_storage} (expected: False)")

            if abs(score_after - score_before) > 1e-7 or abs(mu_norm_after - mu_norm_before) > 1e-7:
                print("  WARNING: deltas are larger than expected (possible dtype/device mismatch).")
            else:
                print("  OK: save/load reproduced probe numbers.")

    # Final summary
    print("\nFinal:")
    print(f"  LinearTS cumulative regret (mistakes): {sum(lin_regrets):.1f}")
    print(f"  NeuralTS cumulative regret (mistakes): {sum(neu_regrets):.1f}")
    print(f"  LinearTS moving avg accuracy (last 200): {moving_avg(lin_rewards, 200):.3f}")
    print(f"  NeuralTS moving avg accuracy (last 200): {moving_avg(neu_rewards, 200):.3f}")

    if args.plot:
        lin_cum: list[float] = []
        neu_cum: list[float] = []
        s1 = 0.0
        s2 = 0.0
        for r1, r2 in zip(lin_regrets, neu_regrets):
            s1 += r1
            s2 += r2
            lin_cum.append(s1)
            neu_cum.append(s2)

        maybe_plot({"LinearTS": lin_cum, "NeuralTS": neu_cum}, title="Cumulative regret (mistakes)")
        maybe_plot({"LinearTS": lin_rewards, "NeuralTS": neu_rewards}, title="Reward / accuracy per step")


if __name__ == "__main__":
    main()
