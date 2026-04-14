# Proof of SSP Algorithms Stability

## Assumptions

- **Proper Policies:** All policies considered are proper, i.e.,
$$
\mathbb{P}_x^\mu(\tau < \infty) = 1, \quad \mathbb{E}_x^\mu[\tau] < \infty, \quad \forall x \in S.
$$

- **Bounded Costs:** There exists $g_{\max} < \infty$ such that
$$
0 \le g(x,u,y) \le g_{\max}, \quad \forall x,u,y.
$$

- **Approximation Error:** For each policy $\mu$, the ALP approximation satisfies
$$
\|J^\mu - J^{\text{ALP}}_\mu\|_\infty \le \beta,
$$
where $\beta > 0$ is bounded.

- **Expected Termination is Bounded:**
$$
\sup_{x \in S} \mathbb{E}_x^{\mu_{t+1}}[\tau] \le H.
$$

---

## Bellman Equation

The Bellman operator for a policy $\mu$ in the SSP setting is:
$$
(T_\mu J)(x) = \sum_{y \in S} P(y \mid x, \mu(x)) \big( g(x,\mu(x),y) + J(y) \big).
$$

It satisfies the fixed-point relation:
$$
J^\mu = T_\mu J^\mu.
$$

---

## Main Result

**Theorem (SSP Analogue of Theorem 2):**

Let $\mu_t$ be the current policy and $\mu_{t+1}$ be the updated policy obtained via decentralized policy improvement using $J^{\text{ALP}}_{\mu_t}$. Then, for all $x \in S$:
$$
J^{\mu_{t+1}}(x) \le J^{\mu_t}(x) + \beta_t H,
$$
where $\beta_t = \|J^{\mu_t} - J^{\text{ALP}}_\mu\|_\infty$.

---

## Proof

Let $\mu := \mu_t$, $\tilde{\mu} := \mu_{t+1}$, and $\beta := \beta_t$.

We begin with one step of the Bellman operator under $\tilde{\mu}$:
$$
(T_{\tilde{\mu}} J^\mu)(x) = \sum_{y \in S} P(y \mid x, \tilde{\mu}(x)) \big( g(x,\tilde{\mu}(x),y) + J^\mu(y) \big).
$$

Using the approximation error bound $J^\mu(y) \le J^{\text{ALP}}_\mu(y) + \beta$:
$$
T_{\tilde{\mu}} J^\mu(x) \le \sum_{y \in S} P(y \mid x, \tilde{\mu}(x)) \big( g(x,\tilde{\mu}(x),y) + J^{\text{ALP}}_\mu(y) \big) + \beta.
$$

Observe that the summation term is exactly $T_{\tilde{\mu}} J^{\text{ALP}}_\mu(x)$. Since $\tilde{\mu}$ is greedy with respect to $J^{\text{ALP}}_\mu$:
$$
T_{\tilde{\mu}} J^{\text{ALP}}_\mu(x) \le T_\mu J^{\text{ALP}}_\mu(x).
$$

From feasibility of the ALP ($J^{\text{ALP}}_\mu(x) \le J^\mu(x)$), we have:
$$
T_\mu J^{\text{ALP}}_\mu(x) \le T_\mu J^\mu(x) = J^\mu(x).
$$

Combining these steps:
$$
T_{\tilde{\mu}} J^\mu(x) \le J^\mu(x) + \beta.
$$

Applying this recursively for the hitting time $\tau$ and taking the expectation:
$$
J^{\tilde{\mu}}(x) = \mathbb{E}_x^{\tilde{\mu}} \big[ T_{\tilde{\mu}}^\tau J^\mu(x) \big] \le J^\mu(x) + \beta \, \mathbb{E}_x^{\tilde{\mu}}[\tau].
$$

Using $\mathbb{E}_x^{\tilde{\mu}}[\tau] \le H$:
$$
J^{\tilde{\mu}}(x) \le J^\mu(x) + H\beta.
$$