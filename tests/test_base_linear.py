import torch

from bandexa.posterior.bayes_linear import BayesianLinearRegression


def test_shapes_and_basic_ops():
    d = 5
    blr = BayesianLinearRegression(dim=d, prior_var=2.0, obs_noise_var=0.5)

    z = torch.randn(d)
    r = torch.tensor(1.25)

    # update single
    blr.update(z, r)

    mu = blr.posterior_mean()
    assert mu.shape == (d,)

    w = blr.sample_weights()
    assert w.shape == (d,)

    W = blr.sample_weights(n_samples=7)
    assert W.shape == (7, d)

    m = blr.mean_score(z)
    s = blr.sample_score(z)
    assert m.shape == ()
    assert s.shape == ()
    assert torch.isfinite(m)
    assert torch.isfinite(s)


def test_posterior_moves_toward_true_weights():
    torch.manual_seed(0)

    d = 4
    n = 400
    w_true = torch.tensor([0.7, -1.2, 0.3, 2.0], dtype=torch.float32)

    Z = torch.randn(n, d)
    noise_std = 0.05
    r = Z @ w_true + noise_std * torch.randn(n)

    blr = BayesianLinearRegression(dim=d, prior_var=10.0, obs_noise_var=noise_std**2)
    prior_diag = blr.precision_matrix().diag().clone()

    blr.update(Z, r)

    mu = blr.posterior_mean()
    # should be close given enough samples + small noise
    assert torch.allclose(mu, w_true, atol=0.15, rtol=0.25)

    post_diag = blr.precision_matrix().diag()
    assert torch.all(post_diag > prior_diag)
