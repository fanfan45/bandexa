# Bandexa
**Bayesian Adaptive Neural Decision Engine for eXploration & Action**

**Disclaimer:** Bandexa is experimental, evolving software. Use at your own risk. It comes with no guarantees of any kind, including correctness, security or suitability for production. See the `LICENSE` and `docs/privacy.md` for more information.

Bandexa is a **PyTorch-native contextual bandits** library focused on **Neural-Linear Thompson Sampling**: learn a neural representation with backprop, while keeping **uncertainty + exploration** tractable via a **Bayesian linear regression** posterior on top of learned features. 

## What are contextual bandits?

A contextual bandit chooses an action $a$ given context $x$, observes a reward $r$, and learns to maximize reward over time (explore vs exploit). In modern practice, “Thompson Sampling with neural models” is often implemented via approximations such as:

- Bootstrapped / ensemble methods (sample a model/head and act greedily)  
- Dropout-as-approximate Bayesian inference (use dropout at inference as uncertainty proxy) 
- Neural Thompson Sampling variants (TS-style exploration using neural tangent kernel)
- Neural-Linear (neural representation + Bayesian linear head for closed-form posterior sampling)

Bandexa’s core emphasis is the **Neural-Linear** family.

## What is Neural-Linear Thompson Sampling?

Neural-Linear TS splits the problem into:

1) **Representation learning** (neural encoder): learn features $z(x,a)$ via SGD / backprop. 
2) **Uncertainty + exploration** (Bayesian linear regression head): keep a conjugate BLR posterior over linear weights for TS, enabling closed-form updates and efficient sampling. 

This design is popular because it preserves **efficient, well-behaved uncertainty** in the final layer while letting the encoder model complex nonlinear structure. It’s also widely studied and used as a strong baseline in deep contextual bandits literature.

## PyTorch-native by design

Bandexa is intentionally **PyTorch-first**: tensors, modules, training loops, and batching are written in idiomatic PyTorch. The primary goal is a clean, reliable implementation for PyTorch workflows (CPU/GPU).

---

## Quickstart (NeuralLinearTS)

At a high level you:

1) Define an **Encoder** `E` that can produce features for an (x, a) pair:
   - `encode(x, a) -> z`
   - and/or `encode_batch(x, actions) -> Z` (recommended for large candidate sets)

2) Create a replay buffer (for training the encoder with supervised loss)

3) Construct a NeuralLinear TS bandit:
   - `bandit = NeuralLinearTS(encoder=E, buffer=..., config=...)`

4) Online loop:
   - `j = bandit.select_action(x, candidate_actions)`
   - observe reward `r`
   - `bandit.update(x, a_j, r)` (Bayesian posterior update + store transition)
   - periodically: `bandit.train_encoder(...)` and `bandit.rebuild_posterior(...)`

### Sketch:

```python
### Pseudocode ###
E = MyEncoder(...)                     # torch.nn.Module
buffer = MemoryReplayBuffer(...)
bandit = NeuralLinearTS(encoder=E, buffer=buffer, config=...)
# simulation
for t in range(T):
    x = get_context()
    A = get_candidate_actions()        # shape (K, act_dim)
    j = bandit.select_action(x, A)
    r = env.step(x, A[j])
    bandit.update(x, A[j], r)

    if t % train_every_n == 0:
        bandit.train_encoder(...)
        bandit.rebuild_posterior(...)
```

## Examples

See the `examples/` directory in this repository for runnable scripts (simulations).
Start with `examples/README.md` (example index + what each script demonstrates).

A key pattern demonstrated in examples is a realistic two-stage system:

candidate generation / retrieval → Thompson re-ranking on the candidate set.

## Documentation

- `docs/architecture.md` — how the pieces fit together
- `docs/development.md` — local dev workflow
- `docs/privacy.md` — privacy posture for the library
- `docs/roadmap.md` — future work / milestones
- `docs/references/` — mathematical background notes (e.g., Bayesian linear regression used by Neural-Linear TS)

## License

MIT License (see LICENSE).

## Attribution (Optional)
Attribution is not required. If you use Bandexa in research or a public project and decide to give credit, you can credit it as: Keyvan Rahmani, *Bandexa* (2026).