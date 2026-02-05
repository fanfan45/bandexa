# Architecture

This document explains how Bandexa’s main components work together.

## Core components

### 1) Encoders (representation learning)

An encoder is a `torch.nn.Module` responsible for producing features:

- `encode(x, a) -> z` where `z` is a 1D feature vector
- `encode_batch(x, actions) -> Z` where `Z` is a 2D matrix of features for many candidate actions

**`encode_batch` matters** for large action sets, where you want to compute context features once and score many actions efficiently (batching + chunking).

### 2) `BayesianLinearRegression` (BLR) posterior

Neural-Linear TS keeps a Bayesian linear regression model on top of learned features:

- reward model: $r \approx w^Tz$ (often Gaussian likelihood).
- maintain posterior over $w$ with conjugate updates.
- Thompson Sampling: sample $w$ from posterior and pick action with max score.

This retains closed-form posterior sampling and computational efficiency, while the encoder handles nonlinear structure.

### 3) Policies

- **Linear TS**: uses fixed features $phi(x,a)$, updates BLR posterior, TS action selection. This is not a part of the package, but examples of it are given in `docs/examples`.
- **`NeuralLinearTS`**: uses encoder features $z(x,a)$, updates BLR posterior in feature space, and periodically trains encoder from replay.

Neural-Linear is a well-studied “strong baseline” approach in deep contextual bandits. 

### 4) Replay buffer (training data for encoder)

The replay buffer stores transitions such as `(x, a, r)` so the encoder can be trained via supervised learning on accumulated data. This separation is intentional: encoder training can run on a slower schedule than posterior updates. 

Currently two types of replay buffer, `MemoryReplayBuffer` (useful for smaller datasets) and `DiskReplayBuffer` (useful for memory intensive datasets) are included. 

## Online loop data flow

A typical loop looks like:

1) **Context** arrives: `x`
2) **Candidate set** prepared: `actions` shape `(K, act_dim)`
3) **Action selection**
   - encoder produces `Z = encode_batch(x, actions)` (optionally chunked)
   - BLR posterior samples `w ~ p(w | data)`
   - scores: `scores = Z @ w`
   - choose `argmax(scores)`
4) **Observe reward** `r`
5) **Update**
   - posterior update with `(z, r)`
   - store `(x, a, r)` in replay buffer
6) Periodically
   - `train_encoder(...)` from replay
   - `rebuild_posterior(...)` so the posterior matches the updated encoder representation

## Large action sets: retrieval → rerank

When the global action pool is huge, a common production pattern is:

- Stage 1: **retrieve** a manageable candidate set using a cheap model (often dot-product in embedding space). Candidate generation is **not** handled by this package. It must be included in the online loop before neural linear TS step. 
- Stage 2: **rerank** candidates with NeuralLinearTS policy (uncertainty-aware).

Bandexa examples demonstrate how the bandit API fits into this two-stage design, even when the “retrieval” stage is synthetic or user-provided (see `docs/examples/README.md`). 

## Determinism and performance

- Use explicit `torch.Generator` streams for reproducible sampling (env vs policy vs training vs candidate sampling).
- Use chunk sizes for scoring and posterior rebuild to control memory and latency.
- Prefer `encode_batch` to avoid repeating context work K times.
