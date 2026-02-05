import torch
import torch.nn as nn
import pytest

from bandexa.buffers.base import (
    MemoryReplayConfig,
    ReplayBufferProtocol,
    make_replay_buffer,
    parse_buffer_config,
)
from bandexa.policies.neural_linear_ts import NeuralLinearTS, NeuralLinearTSConfig


class ConcatLinearEncoder(nn.Module):
    """
    Simple trainable encoder for tests:
      z = W [context; action]

    Implements both encode() and encode_batch() and accepts **kwargs passthrough.
    """

    def __init__(self, ctx_dim: int, act_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.feature_dim = feature_dim
        self.net = nn.Linear(ctx_dim + act_dim, feature_dim)

        self.last_kwargs: dict[str, object] | None = None

    def encode(self, context: torch.Tensor, action: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.last_kwargs = dict(kwargs)
        x = torch.cat([context, action], dim=-1)
        return self.net(x)

    def encode_batch(self, context: torch.Tensor, actions: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.last_kwargs = dict(kwargs)
        ctx = context.unsqueeze(0).expand(actions.shape[0], -1)
        x = torch.cat([ctx, actions], dim=-1)
        return self.net(x)


def _make_agent(
    *,
    encoder: nn.Module,
    feature_dim: int,
    prior_var: float,
    obs_noise_var: float,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    buffer_capacity: int,
    encode_kwargs: dict[str, object] | None = None,
) -> NeuralLinearTS:
    """
    Construct a NeuralLinearTS in the new config-driven style.
    """
    cfg = NeuralLinearTSConfig(
        feature_dim=int(feature_dim),
        posterior_type="bayes_linear",
        prior_var=float(prior_var),
        obs_noise_var=float(obs_noise_var),
        lr=float(lr),
        buffer=MemoryReplayConfig(capacity=int(buffer_capacity)),
        encode_kwargs=dict(encode_kwargs or {}),
    )
    return NeuralLinearTS(
        encoder=encoder,
        config=cfg,
        device=device,
        dtype=dtype,
    )


def test_neural_linear_ts_end_to_end_smoke():
    torch.manual_seed(0)

    ctx_dim = 6
    act_dim = 4
    feature_dim = 8

    device = torch.device("cpu")
    dtype = torch.float32

    encoder = ConcatLinearEncoder(ctx_dim, act_dim, feature_dim)

    bandit = _make_agent(
        encoder=encoder,
        feature_dim=feature_dim,
        prior_var=5.0,
        obs_noise_var=0.25,
        lr=1e-2,
        device=device,
        dtype=dtype,
        buffer_capacity=10_000,
        encode_kwargs={"mode": "default"},
    )

    # Ground-truth reward model for the *environment* (not the bandit posterior)
    w_env = torch.randn(ctx_dim + act_dim)

    def env_reward(context: torch.Tensor, action: torch.Tensor) -> float:
        x = torch.cat([context, action], dim=-1)
        mean = float(x @ w_env)
        p = 1.0 / (1.0 + torch.exp(torch.tensor(-mean))).item()
        return float(torch.bernoulli(torch.tensor(p)).item())

    n_steps = 200
    n_actions = 12

    cb_calls = {"count": 0}

    def cb(step: int, metrics: dict[str, float]) -> None:
        cb_calls["count"] += 1
        assert "loss" in metrics
        assert "buffer_size" in metrics

    for t in range(n_steps):
        context = torch.randn(ctx_dim)
        actions = torch.randn(n_actions, act_dim)

        # test encode_kwargs passthrough override
        a_idx = bandit.select_action(context, actions, encode_kwargs={"t": t}, chunk_size=32)
        assert 0 <= a_idx < n_actions
        assert encoder.last_kwargs is not None and encoder.last_kwargs.get("t") == t

        action = actions[a_idx]
        r = env_reward(context, action)

        bandit.update(context, action, r, encode_kwargs={"t": t})
        assert encoder.last_kwargs is not None and encoder.last_kwargs.get("t") == t

        # occasionally train + rebuild
        if (t + 1) % 50 == 0:
            last_loss = bandit.train_encoder(
                optimizer_steps=10,
                batch_size=32,
                callback=cb,
                log_every=5,
            )
            assert torch.isfinite(torch.tensor(last_loss))

            bandit.rebuild_posterior(chunk_size=64)

            # Flush should be safe even if it's a no-op for memory buffers.
            if hasattr(bandit, "flush_buffer"):
                bandit.flush_buffer()

    assert cb_calls["count"] > 0


def test_select_action_chunking_path_matches_non_chunked():
    """
    Force the chunked path in select_action() and ensure it is consistent with
    the non-chunked path under identical RNG state and deterministic encoder.

    This guards against future regressions in the chunking logic.
    """
    torch.manual_seed(123)

    ctx_dim = 5
    act_dim = 3
    feature_dim = 7

    device = torch.device("cpu")
    dtype = torch.float32

    encoder = ConcatLinearEncoder(ctx_dim, act_dim, feature_dim)
    bandit = _make_agent(
        encoder=encoder,
        feature_dim=feature_dim,
        prior_var=2.0,
        obs_noise_var=1.0,
        lr=1e-2,
        device=device,
        dtype=dtype,
        buffer_capacity=1000,
    )

    # put some signal into the posterior so sampling isn't exactly prior-only
    for _ in range(50):
        c = torch.randn(ctx_dim)
        a = torch.randn(act_dim)
        r = float(torch.randn(()).item())
        bandit.update(c, a, r)

    context = torch.randn(ctx_dim)
    actions = torch.randn(257, act_dim)  # not divisible by chunk size

    # Use explicit generators so both calls sample identical posterior weights
    g1 = torch.Generator(device=device).manual_seed(999)
    g2 = torch.Generator(device=device).manual_seed(999)

    idx_full = bandit.select_action(context, actions, generator=g1, chunk_size=None)
    idx_chunked = bandit.select_action(context, actions, generator=g2, chunk_size=32)

    assert 0 <= idx_full < actions.shape[0]
    assert 0 <= idx_chunked < actions.shape[0]
    assert idx_full == idx_chunked


# -----------------------------------------------------------------------------
# New essential tests for the buffer config + memory replay semantics
# -----------------------------------------------------------------------------


def test_parse_buffer_config_memory_and_disk():
    mem = parse_buffer_config({"kind": "memory", "capacity": 123})
    assert getattr(mem, "kind", None) == "memory"
    assert int(getattr(mem, "capacity")) == 123

    disk = parse_buffer_config({"kind": "disk", "root_dir": "/tmp/x", "capacity": 999})
    assert getattr(disk, "kind", None) == "disk"
    assert int(getattr(disk, "capacity")) == 999
    assert str(getattr(disk, "root_dir")) == "/tmp/x"


def test_memory_replay_all_and_iter_batches_match_chronological_order_no_wrap():
    """
    MemoryReplayBuffer guarantees:
      - all() returns chronological order (oldest -> newest)
      - iter_batches(shuffle=False) yields the same chronological order, chunked
    """
    device = torch.device("cpu")
    dtype = torch.float32

    buf = make_replay_buffer(MemoryReplayConfig(capacity=10), device=device, dtype=dtype)
    assert isinstance(buf, ReplayBufferProtocol)
    assert hasattr(buf, "iter_batches")
    assert hasattr(buf, "all")

    # rewards encode insertion order
    for i in range(7):
        ctx = torch.full((3,), float(i), device=device, dtype=dtype)
        act = torch.full((2,), float(i), device=device, dtype=dtype)
        buf.add(ctx, act, float(i))

    batch_all = buf.all()
    rewards_all = [float(x) for x in batch_all.rewards.detach().cpu().tolist()]
    assert rewards_all == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    rewards_stream: list[float] = []
    for b in buf.iter_batches(batch_size=3, shuffle=False, generator=None):  # type: ignore[attr-defined]
        rewards_stream.extend([float(x) for x in b.rewards.detach().cpu().tolist()])

    assert rewards_stream == rewards_all


def test_memory_replay_chronological_order_with_wraparound():
    """
    When the ring wraps, chronological order should be oldest->newest of the retained window.
    """
    device = torch.device("cpu")
    dtype = torch.float32

    cap = 5
    buf = make_replay_buffer(MemoryReplayConfig(capacity=cap), device=device, dtype=dtype)

    # Add 8 samples into capacity 5 => retain [3,4,5,6,7]
    for i in range(8):
        ctx = torch.full((1,), float(i), device=device, dtype=dtype)
        act = torch.full((1,), float(i), device=device, dtype=dtype)
        buf.add(ctx, act, float(i))

    assert len(buf) == cap

    batch_all = buf.all()
    rewards_all = [float(x) for x in batch_all.rewards.detach().cpu().tolist()]
    assert rewards_all == [3.0, 4.0, 5.0, 6.0, 7.0]

    rewards_stream: list[float] = []
    for b in buf.iter_batches(batch_size=2, shuffle=False, generator=None):  # type: ignore[attr-defined]
        rewards_stream.extend([float(x) for x in b.rewards.detach().cpu().tolist()])
    assert rewards_stream == rewards_all


def test_memory_replay_iter_batches_shuffle_is_permutation_and_accepts_extra_kwargs():
    """
    MemoryReplayBuffer.iter_batches:
      - supports shuffle=True with a generator
      - accepts **kwargs (e.g. include_current=True) without failing
    """
    device = torch.device("cpu")
    dtype = torch.float32

    buf = make_replay_buffer(MemoryReplayConfig(capacity=10), device=device, dtype=dtype)

    for i in range(10):
        ctx = torch.zeros(2, device=device, dtype=dtype)
        act = torch.zeros(3, device=device, dtype=dtype)
        buf.add(ctx, act, float(i))

    g = torch.Generator(device=device).manual_seed(123)

    rewards_shuf: list[float] = []
    for b in buf.iter_batches(  # type: ignore[attr-defined]
        batch_size=4,
        shuffle=True,
        generator=g,
        include_current=True,  # tolerated extra kwarg
    ):
        rewards_shuf.extend([float(x) for x in b.rewards.detach().cpu().tolist()])

    assert sorted(rewards_shuf) == [float(i) for i in range(10)]


def test_memory_replay_enforces_consistent_shapes():
    device = torch.device("cpu")
    dtype = torch.float32
    buf = make_replay_buffer(MemoryReplayConfig(capacity=10), device=device, dtype=dtype)

    buf.add(torch.zeros(3), torch.zeros(2), 0.0)

    # context shape mismatch
    with pytest.raises(ValueError):
        buf.add(torch.zeros(4), torch.zeros(2), 0.0)

    # action shape mismatch (context shape ok)
    with pytest.raises(ValueError):
        buf.add(torch.zeros(3), torch.zeros(5), 0.0)
