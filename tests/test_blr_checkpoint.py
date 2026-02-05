import torch

from bandexa.posterior.bayes_linear import BayesianLinearRegression

"""BLR round-trip correctness"""

def test_blr_state_dict_round_trip_cpu():
    torch.manual_seed(0)
    dim = 7

    blr = BayesianLinearRegression(dim=dim, prior_var=2.0, obs_noise_var=0.5, jitter=1e-6)
    # do a few updates
    for _ in range(5):
        z = torch.randn(dim)
        r = float(torch.randn(()).item())
        blr.update(z, r)

    # capture some outputs
    mu_before = blr.posterior_mean().detach().clone()
    Lambda_before = blr.precision_matrix().detach().clone()

    # round-trip
    state = blr.state_dict()
    blr2 = BayesianLinearRegression(dim=dim, prior_var=2.0, obs_noise_var=0.5, jitter=1e-6)
    blr2.load_state_dict(state)

    mu_after = blr2.posterior_mean().detach()
    Lambda_after = blr2.precision_matrix().detach()

    assert torch.allclose(mu_before, mu_after, atol=0, rtol=0)
    assert torch.allclose(Lambda_before, Lambda_after, atol=0, rtol=0)


def test_blr_state_dict_dim_mismatch_raises():
    blr = BayesianLinearRegression(dim=3, prior_var=1.0, obs_noise_var=1.0)
    state = blr.state_dict()

    blr_wrong = BayesianLinearRegression(dim=4, prior_var=1.0, obs_noise_var=1.0)
    try:
        blr_wrong.load_state_dict(state)
        assert False, "Expected ValueError due to dim mismatch"
    except ValueError:
        pass
