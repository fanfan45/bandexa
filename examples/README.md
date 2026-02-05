# Examples

These examples are meant to be **copy-paste runnable** after installing bandexa.

```pip install bandexa``` and then copy the example script or notebook locally and run it as explained below.


## 01 — Synthetic regret (fully online loop)

**File:** `examples/01_synthetic_regret.py`

### What it shows

A fully online contextual bandit loop:

select action → observe reward → update posterior → (periodic) train encoder → rebuild posterior

It prints:

- **Cumulative regret** (computed against the *expected* reward under the synthetic environment)
- **Moving-average reward** (computed from the *observed* noisy rewards)

### Compares

- **LinearTS**: fixed features ϕ(x, a) = concat(context, action) + Bayesian linear regression posterior
- **NeuralLinearTS**: learned features z = encoder([context; action]) + Bayesian linear regression posterior

This is the most portable example because it has no external data.

---

### Run

From the repo root:

```bash
# See all knobs/options
python examples/01_synthetic_regret.py --help

# default
python examples/01_synthetic_regret.py

# Run longer (makes difference clearer)
# A longer horizon often makes the NeuralLinearTS advantage show up more clearly.
python examples/01_synthetic_regret.py --horizon 4000

# Make the environment more nonlinear
# Increasing --interaction-scale strengthens the (context ⊙ action) 
# interaction weights in the environment, which typically increases the gap 
# between NeuralLinearTS and LinearTS.
python examples/01_synthetic_regret.py --interaction-scale 1.5

# Add --plot to visualize regret curves (requires matplotlib)
python examples/01_synthetic_regret.py --horizon 4000 --interaction-scale 1.5 --plot

# If you have a GPU:
python examples/01_synthetic_regret.py --device cuda
```

## 02 — MNIST bandit (image context)

**File:** examples/02_mnist_bandit.py 

### What it shows
It illustrates a contextual bandit on real, high-dimensional inputs (images). Each step draws an MNIST image x (the context). The actions/arms are the 10 digit labels {0..9}. The bandit picks a label a, observes bandit feedback only for that chosen label: `reward = 1 if a == y else 0`.

Over time, it should converge toward picking the correct digit more often (higher moving-average accuracy, lower cumulative regret/mistakes).

Because images are high-dimensional, a good encoded representation can make a big difference. This example compares:
- LinearTS (fixed features): a linear “classifier-like” feature map over pixels and the chosen action.
- NeuralLinearTS (learned features): learns a joint embedding z = φ(image, action) and runs Thompson sampling in that learned feature space.

### Image + buffer handling:

- LinearTS: always uses the original MNIST tensor (1, 28, 28) so the fixed-feature dimension stays tractable.
- NeuralLinearTS: with the ConvNet encoder also uses the original MNIST tensor (1, 28, 28).
- NeuralLinearTS: with a ResNet encoder uses a ResNet-compatible transform: convert (1, 28, 28) → (3, 224, 224) and apply ImageNet normalization, since pretrained ResNet expects that input format.

Replay buffer backend: for ResNet runs, the context tensors are much larger, so the replay buffer defaults to disk-backed replay (recommended) to avoid large RAM usage; for ConvNet runs, an in-memory buffer is typically fine.

### Demonstrated encoders

Default: small ConvNet (recommended baseline)
Fast and self-contained. This is the best starting point for most users.

Optional: pretrained ResNet backbone (parameter-efficient bandit on top of a mostly-frozen vision model)
You can switch to `--encoder resnet18` (or resnet34/50/101/152). By default the script:
- downloads pretrained weights (unless --no-pretrained option is set)
- freezes most ResNet backbone weights and unfreezes only the last residual stage (layer4) for lightweight fine-tuning

In this example we used a parallel action path, which is ResNet-friendly action conditioning. A pretrained vision model like ResNet expects only an image input. We don’t want to rewrite its architecture to accept concatenated <image, action>. Instead, the encoder is built in a two-path (two-tower-like) way:
- backbone(image) -> h_img
- action_dense(one_hot_action) -> h_act (trainable)
- fuse([h_img; h_act]) -> z (trainable)

This is parameter-efficient method to train the contextual bandit. With a mostly frozen backbone, you still learn how actions interact with the image representation via small trainable layers (trainable layers + action_dense + fuse).

Note about using ResNet: 

The primary goal of using ResNet in this example is to show how to use a large pretrained model in Bandexa, not to demonstrate the performance. Pretrained ResNet features are already strong for many visual signals. As a result, fine-tuning even a small part of a large pretrained backbone (e.g., only layer4) often yields incremental and sometimes noisy gains over a few thousand steps—especially on CPU and when using a disk-backed replay buffer. You should still see that NeuralLinearTS is competitive and often improves regret/accuracy compared to LinearTS, but don’t expect ImageNet-style “training curves.” If you want larger gains, run longer horizons and feel free to edit the backbone fine-tuning policy in `examples/_resnet_backbone.py`.

### Run
```bash
# Show all knobs/options
python examples/02_mnist_bandit.py --help

# Default (small convnet)
python examples/02_mnist_bandit.py

# Optional: plot curves (requires matplotlib)
python examples/02_mnist_bandit.py --plot

# ResNet encoder (pretrained weights downloaded by default, backbone most layers frozen)
python examples/02_mnist_bandit.py --encoder resnet18

# If you have CUDA
python examples/02_mnist_bandit.py --encoder resnet18 --device cuda
```
### Tips
#### Warmup exploration

Warm up exploration matters (especially for the small convnet). Bandits can get stuck early if they exploit too soon. Use a random-action warmup so the replay buffer has diversity:

`--warmup-random 1000` (default in the example)
If you set this too low, results can look unstable or flat.

#### Encoder choice

`--encoder convnet` (default): fastest, no extra download.

`--encoder <resnet18|34|50|101|152>`: heavier, but demonstrates “pretrained features + small action-conditioning head” (with optional light backbone fine-tuning via the default layer4 unfreeze).

`--no-pretrained`: don’t download ImageNet weights (generally not recommended for the ResNet path).

#### Buffer choice

`--buffer-backend auto` (default): picks the sensible backend for you.
- convnet: memory replay (contexts are small: (1,28,28))
- resnet*: disk replay (contexts are large: (3,224,224) after ResNet preprocessing).

`--buffer-backend memory`: always use an in-memory replay buffer (fastest, but RAM-heavy for ResNet).

`--buffer-backend disk`: always use a disk-backed replay buffer (recommended for ResNet; scalable).

Optional disk knobs (only relevant when backend is disk):

`--buffer-dir <path>`: where replay shards are stored.

`--disk-shard-size <N>`: samples per shard file.

`--disk-cache-shards <K>`: how many shards to keep cached for sampling.

`--flush-every <N>`: flush partial shard every N steps (0 disables).

#### Training cadence

`--train-every`: how often to update encoder weights from replay

`--optimizer-steps`, `--batch-size`: how hard each training burst is

`--rebuild-chunk`: how many logged samples per chunk when rebuilding the posterior

#### Freezing strategy (ResNet path)
By default we freeze most backbone weights for speed and stability, and train only the last residual stage (layer4). This gives a practical “best of both worlds”: strong pretrained features with a small amount of task adaptation. If you want a different policy (freeze everything, train everything, or unfreeze a different subset), edit the backbone helper `examples/_resnet_backbone.py` to implement your preferred strategy. This tweak is kept out of the example CLI so the example stays simple.

#### Save/load sanity check

`--sanity-check-save-load` saves an inference checkpoint at the end of the run, reloads into a fresh agent, and prints a small numeric probe to confirm the checkpoint round-trip.

`--save-path` controls where the checkpoint goes.

## 03 — Large action space + candidate generation

**File:** `examples/03_two_tower_synthetic.py`

### What it shows

A large-action-set contextual bandit loop with a realistic **two-stage** structure:

1) **Candidate generation / retrieval** (cheap)  
2) **Thompson Sampling re-ranking** on the candidate subset (more expensive, but only over *M* candidates)

This example is intentionally lightweight:

- Pure synthetic tensors, runs on CPU or GPU
- Focus is on **mechanics**: large action pools, `encode_batch()`, chunking, and the online bandit loop

It prints:

- **Cumulative expected regret** (computed against the environment’s *expected* reward on the candidate set)
- **Moving-average reward** (computed from observed noisy rewards)
- **Moving-average expected reward diagnostics**:
  - `ma_best_p` (oracle best expected reward among candidates)
  - `ma_p_lin`, `ma_p_neu` (expected reward of the chosen action for each agent)

### Compares

- **LinearTS**: fixed features ϕ(x, a) = concat(context, action) + Bayesian linear regression posterior
- **NeuralLinearTS**: learned features z = two-tower encoder([context; action]) + Bayesian linear regression posterior

The two-tower encoder uses:

- `u = f_ctx(x)`
- `v = f_act(a)`
- `z = [u; v; u*v]` (elementwise product as an interaction feature)

The key efficiency point is `encode_batch()`:

- compute `u` once per step
- compute `v` for all candidates
- build `z` for all candidates without repeating the context tower work

---

### Run

From the repo root:

```bash
# See all knobs/options
python examples/03_two_tower_synthetic.py --help

# Default
python examples/03_two_tower_synthetic.py

# Larger horizon (more stable curves)
python examples/03_two_tower_synthetic.py --horizon 5000

# Large action pool with a candidate subset (typical large-action setting)
python examples/03_two_tower_synthetic.py --n-actions 10000 --candidate-size 512 --horizon 5000

# Faster training cadence for the encoder (more frequent updates)
python examples/03_two_tower_synthetic.py --candidate-size 512 --train-every 100 --optimizer-steps 50 --horizon 5000

# If you have a GPU:
python examples/03_two_tower_synthetic.py --device cuda
```

### Tips

Regret is computed over the candidate set (by design).
This example is modeling the common production pattern “retrieval → rerank”.
The script computes `best_p` as the best expected reward among the candidate actions only, and regret is `best_p - chosen_p` within that same candidate set. This keeps the oracle cheap and the metric aligned with the reranking stage.

Candidate generation here is a stand-in for a real retrieval system.
In production you might use: 
- an ANN index
- a separate retrieval model
- heuristics or business rules
- a cached embedding index, etc.

This example’s candidate generator exists to demonstrate the interface and mechanics: you pass a candidate subset into select_action().

#### Observed reward vs expected reward
The script prints both:

`ma_reward`: moving average of the sampled/noisy reward

`ma_p_*`: moving average of the expected reward of the chosen action (p_lin, p_neu)
If you want to judge “who is actually picking better actions”, the expected metrics (ma_p_lin, ma_p_neu) are the cleanest signal.

The gap you see in the outputs may be “visible but not dramatic” because several factors naturally compress the headroom:
- Candidate-oracle ceiling: once retrieval/candidates restrict the action set, both agents are optimizing within a smaller space
- Posterior model is Bayesian linear regression: the TS head is linear in the feature vector it sees; the encoder’s job is to make the problem “more linear” in that feature space
- Online learning + warmup exploration: early on, both agents intentionally explore; the gap typically becomes clearer after encoder training/rebuild cycles
- Stochasticity/noise: even with Gaussian noise, per-step rewards fluctuate; the expected metrics smooth that out

It counts as “working” for this example if you see stable, reproducible runs with a fixed seed where: 
- NeuralTS having lower cumulative expected regret than LinearTS over time.
- `ma_p_neu` typically above `ma_p_lin` once learning kicks in.
