import os
import torch
import torch.nn as nn

from bandexa.buffers.base import MemoryReplayConfig
from bandexa.policies.neural_linear_ts import NeuralLinearTS, NeuralLinearTSConfig

"""NeuralLinearTS save/load inference (no replay buffer needed)"""


class TinyEncoder(nn.Module):
    """Simple deterministic encoder with encode + encode_batch."""
    def __init__(self, ctx_dim: int, act_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + act_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
        )

    def encode(self, context: torch.Tensor, action: torch.Tensor, **kwargs: object) -> torch.Tensor:
        x = torch.cat([context, action], dim=-1)
        return self.net(x)

    def encode_batch(self, context: torch.Tensor, actions: torch.Tensor, **kwargs: object) -> torch.Tensor:
        ctx = context.unsqueeze(0).expand(actions.shape[0], -1)
        x = torch.cat([ctx, actions], dim=-1)
        return self.net(x)


def _make_agent(device: torch.device) -> NeuralLinearTS:
    ctx_dim = 5
    act_dim = 3
    feature_dim = 11
    enc = TinyEncoder(ctx_dim, act_dim, feature_dim).to(device=device, dtype=torch.float32)

    cfg = NeuralLinearTSConfig(
        feature_dim=feature_dim,
        posterior_type="bayes_linear",
        prior_var=2.0,
        obs_noise_var=0.7,
        lr=1e-3,
        buffer=MemoryReplayConfig(capacity=1000),
        action_chunk_size=2,
        encode_kwargs={"foo": 123},
    )

    agent = NeuralLinearTS(
        encoder=enc,
        config=cfg,
        device=device,
        dtype=torch.float32,
    )
    return agent


def test_neuralts_save_load_inference_round_trip_cpu(tmp_path):
    device = torch.device("cpu")
    torch.manual_seed(0)

    agent = _make_agent(device)

    # Create a tiny "world"
    ctx = torch.randn(5, device=device)
    actions = torch.randn(4, 3, device=device)

    # Drive some updates so posterior isn't just prior
    for t in range(10):
        gen = torch.Generator(device=device).manual_seed(100 + t)
        a_idx = agent.select_action(ctx, actions, generator=gen)
        r = float((a_idx % 2) == 0)  # deterministic reward pattern
        agent.update(ctx, actions[a_idx], r)

    # Reference outputs
    ref_mu = agent.posterior.posterior_mean().detach().clone()
    ref_choice = agent.select_action(
        ctx, actions, generator=torch.Generator(device=device).manual_seed(999)
    )

    # Save
    ckpt_path = tmp_path / "neuralts.pt"
    agent.save_inference(str(ckpt_path))
    assert os.path.exists(ckpt_path)

    # Rebuild a fresh agent with the same architecture
    agent2 = _make_agent(device)
    agent2.load_inference(str(ckpt_path), map_location="cpu")

    # Check metadata (now lives in config)
    assert agent2.feature_dim == agent.feature_dim
    assert agent2.config.action_chunk_size == agent.config.action_chunk_size
    assert agent2.config.encode_kwargs == agent.config.encode_kwargs

    # Check posterior recovered
    mu2 = agent2.posterior.posterior_mean().detach()
    assert torch.allclose(ref_mu, mu2, atol=0, rtol=0)

    # Check TS selection reproducible given same generator seed
    choice2 = agent2.select_action(
        ctx, actions, generator=torch.Generator(device=device).manual_seed(999)
    )
    assert ref_choice == choice2


def test_neuralts_cpu_checkpoint_loads_on_cuda_if_available(tmp_path):
    """CPU → CUDA load (only if CUDA exists)"""
    if not torch.cuda.is_available():
        return

    # save on CPU
    cpu = torch.device("cpu")
    torch.manual_seed(0)
    agent = _make_agent(cpu)

    ctx = torch.randn(5)
    actions = torch.randn(4, 3)
    for t in range(5):
        gen = torch.Generator(device=cpu).manual_seed(200 + t)
        a_idx = agent.select_action(ctx, actions, generator=gen)
        agent.update(ctx, actions[a_idx], float(a_idx == 0))

    ckpt_path = tmp_path / "neuralts_cpu.pt"
    agent.save_inference(str(ckpt_path))

    # load into CUDA-constructed agent
    cuda = torch.device("cuda")
    agent_cuda = _make_agent(cuda)
    agent_cuda.load_inference(str(ckpt_path), map_location="cpu")

    # run a simple call to ensure everything moved/works
    ctx_cuda = ctx.to(cuda)
    actions_cuda = actions.to(cuda)
    gen_cuda = torch.Generator(device=cuda).manual_seed(777)
    _ = agent_cuda.select_action(ctx_cuda, actions_cuda, generator=gen_cuda)


def test_neuralts_feature_dim_mismatch_raises(tmp_path):
    ckpt = tmp_path / "x.pt"

    agent = _make_agent(torch.device("cpu"))
    agent.save_inference(str(ckpt))

    # different feature_dim agent
    class OtherEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 9)

        def encode(self, context, action, **kwargs):
            return self.lin(torch.cat([context, action], dim=-1))

        def encode_batch(self, context, actions, **kwargs):
            ctx = context.unsqueeze(0).expand(actions.shape[0], -1)
            return self.lin(torch.cat([ctx, actions], dim=-1))

    enc = OtherEncoder()

    cfg_bad = NeuralLinearTSConfig(
        feature_dim=9,  # mismatch vs checkpoint 11
        posterior_type="bayes_linear",
        prior_var=2.0,
        obs_noise_var=0.7,
        lr=1e-3,
        buffer=MemoryReplayConfig(capacity=1000),
    )

    agent_bad = NeuralLinearTS(
        encoder=enc,
        config=cfg_bad,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    try:
        agent_bad.load_inference(str(ckpt), map_location="cpu")
        assert False, "Expected ValueError due to feature_dim mismatch"
    except ValueError:
        pass
