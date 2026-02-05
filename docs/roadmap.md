# Roadmap

**Design goal:** keep the library small and easy to understand. Users should be able to plug in an **encoder model** and a **candidate generator** and get going quickly. Bandexa should provide clear protocols/interfaces for what can be plugged in (encoders, buffers, candidate generators), but avoid prescribing heavy implementations.

## Near-term

- Examples showcasing separate processes for action selection and training loops

## Medium-term

- Add additional posterior families:
  - Bayesian logistic / Bernoulli rewards (non-conjugate; approximation required)
  - alternative noise models / priors

## Longer-term

- Additional policies beyond Neural-Linear TS:
  - bootstrapped heads / ensembles
  - dropout-based uncertainty
  - neural TS variants
- Benchmark harness (small, synthetic + lightweight public datasets)
