# Proof of SSP Algorithms stability

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
  for some $\beta > 0$ and $\beta < \infty$ is bounded.

- **Expected Termination is Bounded:**
  $$
  \sup_{x \in S} \mathbb{E}_x^{\mu_{t+1}} [\tau] \leq H.
  $$

---

## Bellman Equation

The Bellman operator for a policy $\mu$ in the SSP setting is given by:
$$
(T_\mu J)(x) = \sum_{y \in S} P(y|x,\mu(x)) \left[ g(x,\mu(x),y) + J(y) \right].
$$

It satisfies the fixed-point relation:
$$
J^\mu = T_\mu J^\mu.
$$

---

## Main Result

**Theorem (SSP Analogue of Theorem 2):**

Let $\mu_t$ be the current policy and $\mu_{t+1}$ be the updated policy obtained via decentralized policy improvement using $J^{\text{ALP}}_{\mu_t}$. Under the stated assumptions, the following bound holds for all $x \in S$:
$$
J^{\mu_{t+1}}(x)
\le
J^{\mu_t}(x)
+
\beta_t \cdot H,
$$
where $\beta_t = \|J^{\mu_t} - J^{\text{ALP}}_{\mu_t}\|_\infty$.

---

## Proof

Let $\mu := \mu_t$, $\tilde{\mu} := \mu_{t+1}$, and $\beta := \beta_t$.

We begin by evaluating one step of the Bellman operator under the updated policy:
$$
(T_{\tilde{\mu}} J^\mu)(x)
=
\sum_{y \in S} P(y|x,\tilde{\mu}(x)) \left[ g(x,\tilde{\mu}(x),y) + J^\mu(y) \right].
$$

Using the approximation error bound,
$$
J^\mu(y) \le J^{\text{ALP}}_\mu(y) + \beta,
$$
we obtain:
$$
T_{\tilde{\mu}} J^\mu(x)
\le
\sum_y P(\cdot)\left[g + J^{\text{ALP}}_\mu(y)\right] + \beta.
$$

Now observe:
$$
T_{\tilde{\mu}} J^{\text{ALP}}_\mu(x)
=
\sum_y P(\cdot)\left[g + J^{\text{ALP}}_\mu(y)\right].
$$

Since $\tilde{\mu}$ is greedy w.r.t. $J_{\mu}^{\text{ALP}}$,
$$
T_{\tilde{\mu}} J^{\text{ALP}}_\mu(x) 
\le
T_{\mu} J^{\text{ALP}}_\mu(x).
$$

Thus,
$$
T_{\tilde{\mu}} J^\mu(x)
\le
T_\mu J^{\text{ALP}}_\mu(x) + \beta.
$$

From the ALP feasibility condition,
$$
J^{\text{ALP}}_\mu(x) \le J^\mu(x), \quad \forall x \in S,
$$
which implies:
$$
T_\mu J^{\text{ALP}}_\mu(x)
\le
T_\mu J^\mu(x)
=
J^\mu(x).
$$

Combining the above inequalities:
$$
T_{\tilde{\mu}} J^\mu(x)
\le
J^\mu(x) + \beta.
$$

Applying this recursively, for any integer $k \ge 1$:
$$
T_{\tilde{\mu}}^k J^\mu(x)
\le
J^\mu(x) + k\beta.
$$

Let $\tau$ denote the hitting time of the terminal state under policy $\tilde{\mu}$. Since $\tilde{\mu}$ is proper, $\tau$ is finite almost surely.

Using the interpretation of repeated Bellman operators:
$$
T_{\tilde{\mu}}^\tau J^\mu(x)
\le
J^\mu(x) + \tau \beta.
$$

Taking expectation with trajectories starting from $x$:
$$
J^{\tilde{\mu}}(x)
=
\mathbb{E}_x^{\tilde{\mu}}\left[T_{\tilde{\mu}}^\tau J^\mu(x)\right]
\le
J^\mu(x) + \beta \cdot \mathbb{E}_x^{\tilde{\mu}}[\tau].
$$

Using the assumption $\mathbb{E}_x^{\tilde{\mu}}[\tau] \le H$, we obtain:
$$
J^{\tilde{\mu}}(x)
\le
J^\mu(x) + H\beta.
$$

This completes the proof.