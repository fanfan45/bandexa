# Bayesian Linear Regression Posterior

This document explains the **same Bayesian linear regression posterior** used by both:

- **Linear Thompson Sampling (LinearTS / LinTS)**: linear model on **fixed** features
- **NeuralLinearTS (NeuralLinear / “Neural Thompson Sampling”)**: linear model on **learned** features from a neural encoder

The posterior math is identical in both. The difference is only **where the feature vector comes from**.

---

## Linear Thompson Sampling

At each time step $t$, we observe a context $x_t$ and choose an action $a_t$ from a candidate set $\mathcal{A}_t$

Assume a fixed feature vector $z_{t,a} = \phi(x_t, a) \in \mathbb{R}^d$, where $\langle x_t, a \rangle$ denotes the concatenated input to the feature map $\phi$

The Gaussian reward observation (likelihood) model for the chosen action $a_t$ is:

$$
r_t = w^\top z_{t,a_t} + \epsilon
= w^\top \phi(x_t,a_t) + \epsilon
\qquad
\epsilon \sim \mathcal{N}(0,\sigma^2)
$$

Define the **score** (predicted mean reward) for any candidate action $a$:

$$
s_{t,a} := w^\top z_{t,a}
$$

Under the Gaussian reward model above, the score is the conditional expectation:

$$
\mathbb{E}[r \mid w, z_{t,a}] = s_{t,a}
$$

Optionally, for click/no-click one may interpret $s_{t,a}$ as a logit and map it to a probability:

$$
p_{t,a} = \sigma(s_{t,a})
$$

A truly Bernoulli/logistic last layer where $r_t \in \{0,1\}$ (click/no-click) typically needs approximations (Laplace/VI), but many Linear Thompson Sampling implementations still use the Gaussian model because the posterior update is closed-form and fast

In Linear Thompson Sampling:

1. Maintain posterior $w \sim \mathcal{N}(\mu,\Sigma)$
2. Sample weights $\tilde w \sim \mathcal{N}(\mu,\Sigma)$
3. Score each candidate action as $\tilde s_{t,a} = \tilde w^\top z_{t,a}$ for all $a \in \mathcal{A}_t$
4. Choose $a_t = \arg\max_{a \in \mathcal{A}_t} \tilde s_{t,a}$
5. Observe $r_t$ for the chosen $a_t$ and update the posterior using $(z_{t,a_t}, r_t)$

In **Neural Linear Thompson Sampling**, instead of a fixed feature vector, we use a trainable encoder:

$$
z_{t,a} = \phi_\theta(x_{t,a}) \in \mathbb{R}^d
$$

with trainable parameters $\theta$

---

## Bayesian model for weights $w$

### Prior

Consider the prior:

$$
w \sim \mathcal{N}(\mu_0,\Sigma_0)
$$

With a common isotropic choice:

$$
\mu_0 = 0
\qquad
\Sigma_0 = \alpha I
$$

where $\alpha > 0$ is the **prior variance**

### Likelihood

The likelihood function (Gaussian regression) is:

$$
p(r_t \mid z_t, w) = \mathcal{N}(r_t;\; w^\top z_t,\; \sigma^2)
$$

where $z_t := z_{t,a_t}$ and $\sigma^2 > 0$ is the **observation noise variance** in the reward model (i.e., $\epsilon \sim \mathcal{N}(0,\sigma^2)$)

**Note:** $\alpha$ and $\sigma^2$ are independent hyperparameters: $\alpha$ controls prior uncertainty over $w$, while $\sigma^2$ controls observation noise and therefore how strongly data updates the posterior

---

## Posterior (closed form, batch view)

Suppose you have $n$ observations $\mathcal{D} = \{(z_i, r_i)\}_{i=1}^n$, where $z_i = \phi(x_i, a_i)$ is the feature vector produced by the feature map $\phi$ for the chosen context–action pair $\langle x_i, a_i\rangle$, and $r_i$ is the observed reward for that choice

Stack:

- $\mathbf{Z} \in \mathbb{R}^{n \times d}$ with rows $z_i^\top$
- $\mathbf{r} \in \mathbb{R}^n$

Assuming independent, equal-variance noise across the $n$ observations:

$$
\mathbf{r} \mid w, \mathbf{Z} \sim \mathcal{N}(\mathbf{Z} w,\ \sigma^2 I)
$$

Then the Bayesian posterior is:

$$
w \mid \mathcal{D} \sim \mathcal{N}(\mu_n,\Sigma_n)
$$

The closed-form posterior update for Bayesian linear regression is:

$$
\Sigma_n^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{Z}
$$

$$
\mu_n = \Sigma_n \left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{r}\right)
$$

**Notation:** We use boldface for stacked vectors and matrices (e.g., $\mathbf{r}$ and $\mathbf{Z}$)

---

## Posterior (online / recursive view)

In an online bandit, the posterior can be updated sequentially. The equivalent **online (recursive) update** for a new observation $(z_n, r_n)$ is:

$$
\Sigma_n^{-1} = \Sigma_{n-1}^{-1} + \frac{1}{\sigma^2} z_n z_n^\top
$$

$$
\mu_n = \Sigma_n\left(\Sigma_{n-1}^{-1}\mu_{n-1} + \frac{1}{\sigma^2} z_n r_n\right)
$$

---

## Precision form and sufficient statistics

Rather than maintaining the covariance $\Sigma$ directly, it is numerically convenient to maintain the **precision** matrix:

$$
\Lambda := \Sigma^{-1}
$$

and also maintain:

$$
b := \Lambda \mu
$$

so the posterior mean is recovered by solving:

$$
\Lambda \mu = b
$$

### Connection to the common $A,b$ notation in bandits

A common convention in bandit references defines:

$$
A_n := \lambda I + \sum_{i=1}^n z_i z_i^\top
\qquad
b_n := \sum_{i=1}^n r_i z_i
$$

and then writes:

$$
\mu_n = A_n^{-1} b_n
\qquad
\Sigma_n = \sigma^2 A_n^{-1}
$$

This is the same Bayesian linear regression posterior, with $\sigma^2$ placed outside the matrix inverse. Under this convention:

$$
\Lambda_n = \Sigma_n^{-1} = \frac{1}{\sigma^2} A_n
$$

So whether you update $\Lambda$ with a $\frac{1}{\sigma^2}$ factor (precision form) or update $A$ without it (common bandit form) is a choice of convention

---

## Online and batch updates in precision form

### Update per observation $(z, r)$

$$
\Lambda \leftarrow \Lambda + \frac{1}{\sigma^2} z z^\top
$$

$$
b \leftarrow b + \frac{1}{\sigma^2} z r
$$

Recover the posterior mean by solving:

$$
\Lambda \mu = b
$$

(In code we never explicitly compute $\Lambda^{-1}$; we solve the linear system)

### Batch update form (minibatch)

For a minibatch $\mathbf{Z} \in \mathbb{R}^{n \times d}$ and $\mathbf{r} \in \mathbb{R}^n$:

$$
\Lambda \leftarrow \Lambda + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{Z}
$$

$$
b \leftarrow b + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{r}
$$

---

## Sampling from the posterior (Cholesky of precision)

We need to sample:

$$
w \sim \mathcal{N}(\mu,\Sigma)
\qquad
\Sigma = \Lambda^{-1}
$$

Compute the Cholesky factor of the precision:

$$
\Lambda = L L^\top
$$

Here $L$ is the Cholesky factor of $\Lambda$. Since $\Lambda$ is symmetric positive definite, its Cholesky decomposition gives a lower-triangular matrix $L$. We use $L$ because it lets us sample from $\mathcal{N}(\mu,\Lambda^{-1})$ efficiently using triangular solves (without explicitly computing a matrix inverse)

Sample $\epsilon \sim \mathcal{N}(0, I)$, then:

$$
w = \mu + L^{-\top} \epsilon
$$

Because:

$$
\mathrm{Cov}(L^{-\top}\epsilon) = L^{-\top} I L^{-1} = (LL^\top)^{-1} = \Lambda^{-1} = \Sigma
$$

### Why add “jitter”

Numerically, $\Lambda$ should be positive definite, but floating point can create near-singularity. We stabilize with:

$$
\Lambda_{\text{stable}} = \Lambda + \varepsilon I
$$

before Cholesky, where $\varepsilon$ is small (e.g., $10^{-6}$)

---

## Encoder update for Neural Linear Thompson Sampling

In both LinearTS and NeuralLinearTS, we update the Bayesian weight distribution using the chosen action’s feature vector $z_t := z_{t,a_t}$. In Neural Linear Thompson Sampling, we additionally train the encoder parameters $\theta$ using logged tuples $(x_t, a_t, r_t)$

Common choices for the loss function:

### (A) MSE regression loss (aligned with Gaussian BLR)

$$
\hat r_t = \mu^\top z_t
$$

$$
\mathcal{L}_{\text{MSE}}(\theta) = (\hat r_t - r_t)^2
$$

### (B) BCE for clicks (likelihood-correct, but mismatched to BLR posterior)

$$
p_t = \sigma(\mu^\top z_t)
$$

$$
\mathcal{L}_{\text{BCE}}(\theta) = -\left[r_t\log p_t + (1-r_t)\log(1-p_t)\right]
$$

---

## Mapping to implementation (`bayes_linear.py`)

Our code maintains:

- `Lambda` = posterior precision $\Lambda$
- `b`      = $\Lambda\mu$

Hyperparameters:

- `prior_var = \alpha` meaning $\Sigma_0 = \alpha I$ so $\Lambda_0 = (1/\alpha)I$
- `noise_var = \sigma^2`



$$
\Lambda \leftarrow \Lambda + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{Z}
$$

$$
b \leftarrow b + \frac{1}{\sigma^2} \mathbf{Z}^\top \mathbf{r}
$$

Posterior mean (solve, do not invert):

$$
\Lambda \mu = b
$$

Sampling:

- compute Cholesky of $\Lambda + \varepsilon I$
- sample $\epsilon \sim \mathcal{N}(0,I)$
- return $w = \mu + L^{-\top}\epsilon$

This posterior sampler is exactly what enables Thompson sampling in LinearTS and NeuralLinearTS
