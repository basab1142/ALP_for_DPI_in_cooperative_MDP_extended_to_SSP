"""
Approximate Linear Programming for Decentralized Policy Iteration
in Cooperative Multi-Agent Markov Decision Processes

Implements Algorithms 1-5 from the paper.
"""

import numpy as np
from scipy.optimize import linprog
import itertools
import matplotlib.pyplot as plt
import pandas as pd



# =============================================================================
# Algorithm 1: Decentralized Policy Improvement (DPIm)
# =============================================================================

def DPIm(i, mu_updated, mu_base, J, transition_probs, costs, state_space, action_spaces, alpha=1.0):
    """
    Decentralized Policy Improvement for agent i

    Args:
        i          : Index of the agent being updated (0-indexed).
        mu_updated : List of updated policies for agents 0..i-1.
                     mu_updated[k] is a dict {state: action} for agent k.
        mu_base    : List of base policies for agents i+1..m-1.
                     mu_base[k] is a dict {state: action} for agent k (k > i).
        J          : Value function, array of shape (n_states,).
        transition_probs : Callable p(x, u_joint) -> array of shape (n_states,)
                           giving transition probabilities from state x under
                           joint action u_joint (tuple of per-agent actions).
        costs      : Callable g(x, y, u_joint) -> float, single-stage cost.
        state_space: List of states.
        action_spaces : List of action spaces, one per agent.
                        action_spaces[k] is a list of valid actions for agent k.
        alpha      : Discount factor (use 1.0 for finite-horizon).

    Returns:
        mu_i_new   : Updated policy for agent i, dict {state: action}.
    """
    m = len(mu_updated) + 1 + len(mu_base)
    mu_i_new = {}

    for x in state_space:
        best_action = None
        best_value = np.inf

        # Build joint action: updated agents 0..i-1, then agent i, then base i+1..m-1
        for ui in action_spaces[i]:
            u_joint = []
            for k in range(i):
                u_joint.append(mu_updated[k][x])
            u_joint.append(ui)
            for k in range(i + 1, m):
                u_joint.append(mu_base[k - i - 1][x])
            u_joint = tuple(u_joint)

            probs = transition_probs(x, u_joint)          # shape (n_states,)
            g_vals = np.array([costs(x, y, u_joint) for y in state_space])
            value = np.dot(probs, g_vals + alpha * J)

            if value < best_value:
                best_value = value
                best_action = ui

        mu_i_new[x] = best_action

    return mu_i_new


# =============================================================================
# Algorithm 2: Calculate Approximate Cost Function (CACFN) — Finite Horizon
# =============================================================================

def CACFN_FH(k, pi, J_next, transition_probs, costs, state_space, Phi, c):
    """
    Approximate policy evaluation via Linear Programming (finite horizon).

    Solves:
        max_{r}  c^T Phi r
        s.t.     Phi r(x) <= sum_y p(x,y | mu_k(x)) [g(x,y,mu_k(x)) + J_next(y)]

    Args:
        k          : Current stage index.
        pi         : Policy sequence, pi[k] is a list of per-agent dicts {state: action}.
        J_next     : Approximate cost at stage k+1, array of shape (n_states,).
        transition_probs : Callable p(x, u_joint) -> array of shape (n_states,).
        costs      : Callable g(x, y, u_joint) -> float.
        state_space: List of states (length n).
        Phi        : Feature matrix, shape (n, d).
        c          : State-relevance weights, shape (n,).

    Returns:
        J_ALP      : Approximate cost at stage k, array of shape (n,).
    """
    n, d = Phi.shape
    mu_k = pi[k]  # list of per-agent policies at stage k

    # Build RHS: b(x) = sum_y p(x, u(x)) [g(x,y,u(x)) + J_next(y)]
    b = np.zeros(n)
    for idx, x in enumerate(state_space):
        u_joint = tuple(mu_k[i][x] for i in range(len(mu_k)))
        probs = transition_probs(x, u_joint)
        g_vals = np.array([costs(x, y, u_joint) for y in state_space])
        b[idx] = np.dot(probs, g_vals + J_next)

    # LP: max c^T Phi r  <=>  min -c^T Phi r
    # s.t. Phi r <= b  (n constraints, d variables)
    c_lp = -Phi.T @ c           # shape (d,)
    A_ub = Phi                   # shape (n, d)
    b_ub = b                     # shape (n,)

    result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * d, method='highs')

    if result.success:
        r_opt = result.x
    else:
        r_opt = np.zeros(d)

    J_ALP = Phi @ r_opt
    return J_ALP


# =============================================================================
# Algorithm 3: DPI using ALP for Finite Horizon CO-MA-MDP
# =============================================================================

def DPI_ALP_FH(N, state_space, action_spaces, transition_probs, costs, terminal_cost,
               Phi, c, base_policy=None):
    """
    Approximate Decentralized Policy Iteration for Finite Horizon CO-MA-MDP.

    Args:
        N              : Horizon length (stages 0 .. N-1).
        state_space    : List of states (length n).
        action_spaces  : List of action spaces per agent (length m).
        transition_probs : Callable p(x, u_joint) -> array (n_states,).
        costs          : Callable g(x, y, u_joint, k) -> float (stage k cost).
        terminal_cost  : Array of shape (n,) — terminal cost g_N.
        Phi            : Feature matrix, shape (n, d).
        c              : State-relevance weights, shape (n,).
        base_policy    : Initial policy (list of length N, each entry a list of
                         m dicts {state: action}). Random init if None.

    Returns:
        optimal_policy : List of length N; optimal_policy[k] is a list of m
                         dicts {state: action}.
        J_values       : List of length N+1 of approximate cost arrays.
    """
    n = len(state_space)
    m = len(action_spaces)

    # Initialise base policy randomly if not provided
    if base_policy is None:
        base_policy = []
        for k in range(N):
            stage_policy = []
            for i in range(m):
                agent_policy = {x: np.random.choice(action_spaces[i]) for x in state_space}
                stage_policy.append(agent_policy)
            base_policy.append(stage_policy)

    # Terminal cost (stage N)
    J_values = [None] * (N + 1)
    J_values[N] = terminal_cost.copy()

    optimal_policy = [None] * N
    trajectory = []   # records mean ALP cost after every inner improvement step

    # Wrap costs to accept stage index
    def costs_k(x, y, u_joint, k=0):
        try:
            return costs(x, y, u_joint, k)
        except TypeError:
            return costs(x, y, u_joint)

    # Backward induction
    for k in range(N - 1, -1, -1):
        pi = base_policy
        J_ALP_pi = CACFN_FH(k, pi, J_values[k + 1], transition_probs,
                             lambda x, y, u: costs_k(x, y, u, k), state_space, Phi, c)


        while True:
            # Decentralised policy improvement
            pi_new_k = []
            mu_updated = []
            for i in range(m):
                mu_base_agents = [pi[k][j] for j in range(i + 1, m)]
                mu_i_new = DPIm(
                    i, mu_updated, mu_base_agents,
                    J_values[k + 1], transition_probs,
                    lambda x, y, u: costs_k(x, y, u, k),
                    state_space, action_spaces, alpha=1.0
                )
                mu_updated.append(mu_i_new)
                pi_new_k.append(mu_i_new)

            # Build updated full policy sequence (only stage k changes)
            pi_new = [pi[s].copy() if s != k else pi_new_k for s in range(N)]

            # Approximate policy evaluation for updated policy
            J_ALP_pi_new = CACFN_FH(k, pi_new, J_values[k + 1], transition_probs,
                                     lambda x, y, u: costs_k(x, y, u, k), state_space, Phi, c)

            # if k == 0:
            r = rollout_fh(pi_new, transition_probs, costs_k, state_space, horizon=N)
            trajectory.append(r)

            # Check convergence: stop when updated cost >= base cost everywhere
            if np.all(J_ALP_pi_new >= J_ALP_pi):
                break

            pi = pi_new
            J_ALP_pi = J_ALP_pi_new

        optimal_policy[k] = pi[k]
        J_values[k] = J_ALP_pi

    return optimal_policy, J_values, trajectory


# =============================================================================
# Algorithm 4: Calculate Approximate Cost Function (CACFN) — Infinite Horizon
# =============================================================================

def CACFN_IH(mu, transition_probs, costs, state_space, Phi, c, alpha):
    """
    Approximate policy evaluation via Linear Programming (infinite horizon).

    Solves:
        max_{r}  c^T Phi r
        s.t.     T_mu(Phi r) >= Phi r
        i.e.     Phi r(x) <= sum_y p(x,y|mu(x)) [g(x,y,mu(x)) + alpha * Phi r(y)]

    Rearranged: (Phi - alpha * P_mu Phi) r <= g_mu
    where P_mu[x,y] = p(x,y|mu(x)), g_mu[x] = sum_y p(x,y|mu(x)) g(x,y,mu(x)).

    Args:
        mu         : Joint policy, list of m dicts {state: action}.
        transition_probs : Callable p(x, u_joint) -> array (n_states,).
        costs      : Callable g(x, y, u_joint) -> float.
        state_space: List of states (length n).
        Phi        : Feature matrix, shape (n, d).
        c          : State-relevance weights, shape (n,).
        alpha      : Discount factor in (0, 1).

    Returns:
        J_ALP      : Approximate cost array of shape (n,).
    """
    n, d = Phi.shape
    m = len(mu)

    P_mu = np.zeros((n, n))
    g_mu = np.zeros(n)

    for idx, x in enumerate(state_space):
        u_joint = tuple(mu[i][x] for i in range(m))
        probs = transition_probs(x, u_joint)
        P_mu[idx] = probs
        g_mu[idx] = np.dot(probs, [costs(x, y, u_joint) for y in state_space])

    # Constraint: (I - alpha P_mu) Phi r <= g_mu
    A_ub = (np.eye(n) - alpha * P_mu) @ Phi   # shape (n, d)
    b_ub = g_mu                                 # shape (n,)
    c_lp = -(Phi.T @ c)                         # minimise negative objective

    result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * d, method='highs')

    if result.success:
        r_opt = result.x
    else:
        r_opt = np.zeros(d)

    return Phi @ r_opt


# =============================================================================
# Algorithm 5: DPI using ALP for Infinite Horizon CO-MA-MDP
# =============================================================================

def DPI_ALP_IH(state_space, action_spaces, transition_probs, costs, Phi, c, alpha,
               base_policy=None, T=100):
    """
    Approximate Decentralized Policy Iteration for Infinite Horizon CO-MA-MDP.

    Args:
        state_space    : List of states (length n).
        action_spaces  : List of action spaces per agent (length m).
        transition_probs : Callable p(x, u_joint) -> array (n_states,).
        costs          : Callable g(x, y, u_joint) -> float.
        Phi            : Feature matrix, shape (n, d).
        c              : State-relevance weights, shape (n,).
        alpha          : Discount factor in (0, 1).
        base_policy    : Initial policy, list of m dicts {state: action}. Random if None.
        T              : Maximum number of outer iterations.

    Returns:
        mu             : Final policy (list of m dicts {state: action}).
        J_ALP          : Approximate value function at convergence, shape (n,).
    """
    n = len(state_space)
    m = len(action_spaces)

    # Initialise base policy
    if base_policy is None:
        mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    else:
        mu = [dict(p) for p in base_policy]

    J_ALP = CACFN_IH(mu, transition_probs, costs, state_space, Phi, c, alpha)
    trajectory = []

    for t in range(T):
        # Decentralised policy improvement
        mu_updated = []
        for i in range(m):
            mu_base_agents = [mu[j] for j in range(i + 1, m)]
            mu_i_new = DPIm(
                i, mu_updated, mu_base_agents,
                J_ALP, transition_probs, costs,
                state_space, action_spaces, alpha
            )
            mu_updated.append(mu_i_new)

        mu_new = mu_updated  # length m

        # Approximate policy evaluation
        J_ALP_new = CACFN_IH(mu_new, transition_probs, costs, state_space, Phi, c, alpha)

        r = rollout_ih(mu_new, transition_probs, costs, state_space, horizon=50, gamma=alpha)
        trajectory.append(r)

        # Convergence check: stop when updated cost >= base cost everywhere
        if np.all(J_ALP_new >= J_ALP):
            mu = mu_new
            J_ALP = J_ALP_new
            print(f"  Converged at iteration {t + 1}")
            break

        mu = mu_new
        J_ALP = J_ALP_new

    return mu, J_ALP, trajectory

# ===================================================
# Exact value iteration for the same grid world
# ===================================================

def exact_finite_horizon(transition_probs, costs, state_space, action_spaces,
                         horizon, terminal_cost, gamma=1.0):
    n_states = len(state_space)
    joint_actions = list(itertools.product(*action_spaces))
    V = terminal_cost.copy()  # V_N
    trajectory = []
    policies = []

    for t in range(horizon-1, -1, -1):
        V_new = np.zeros(n_states)
        policy_t = extract_greedy_policy(V, state_space, action_spaces, transition_probs, costs, gamma=1.0)

        policies.insert(0, policy_t)

        for x in state_space:
            best = np.inf

            for u in joint_actions:
                probs = transition_probs(x, u)
                imm_cost = sum(probs[y] * costs(x, y, u) for y in state_space)
                future = sum(probs[y] * V[y] for y in state_space)
                total = imm_cost + gamma * future
                if total < best:
                    best = total
            V_new[x] = best

        # build full policy (same policy for all stages for simplicity)
        # full_policy = [policy_t for _ in range(horizon)]

        # 🔥 rollout
        full_policy = policies + [policies[-1]] * (horizon - len(policies))
        r = rollout_fh(full_policy,
                       transition_probs, costs,
                       state_space, horizon)
        trajectory.append(r)

        V = V_new

    trajectory.reverse()
    return V, trajectory

def exact_infinite_horizon(transition_probs, costs, state_space, action_spaces,
                           gamma, tol=1e-8, max_iter=10000, verbose=True):
    n_states = len(state_space)
    joint_actions = list(itertools.product(*action_spaces))
    # Precompute for each joint action: transition matrix (n x n) and cost vector (n)
    P = {}
    g = {}
    for u in joint_actions:
        P_u = np.zeros((n_states, n_states))
        g_u = np.zeros(n_states)
        for x in state_space:
            probs = transition_probs(x, u)
            P_u[x] = probs
            g_u[x] = sum(probs[y] * costs(x, y, u) for y in state_space)
        P[u] = P_u
        g[u] = g_u

    V = np.zeros(n_states)
    trajectory = []   # record initial (zero) cost

    for it in range(max_iter):
        V_new = np.zeros(n_states)

        for x in state_space:
            best = np.inf

            for u in joint_actions:
                val = g[u][x] + gamma * P[u][x] @ V

                if val < best:
                    best = val

            V_new[x] = best

        # extract policy + rollout
        mu = extract_greedy_policy(V_new, state_space, action_spaces,
                                  transition_probs, costs, gamma)

        r = rollout_ih(mu, transition_probs, costs,
                       state_space, horizon=50, gamma=gamma)
        trajectory.append(r)
        diff = np.max(np.abs(V_new - V))
        V = V_new

        if verbose and (it+1) % 1000 == 0:
            print(f"Iter {it+1}, diff={diff:.2e}")
        if diff < tol:
            if verbose:
                print(f"Converged after {it+1} iterations, diff={diff:.2e}")
            break
    else:
        print(f"Warning: did not converge after {max_iter} iterations, diff={diff:.2e}")

    return V, trajectory


def evaluate_policy_fh(policy, transition_probs, costs,
                       state_space, horizon, terminal_cost):
    """
    Policy evaluation via backward DP
    """
    n_states = len(state_space)
    V = terminal_cost.copy()

    for t in range(horizon-1, -1, -1):
        V_new = np.zeros(n_states)
        mu_t = policy[t]

        for x in state_space:
            u = tuple(mu_t[i][x] for i in range(len(mu_t)))
            probs = transition_probs(x, u)

            val = sum(probs[y] * costs(x, y, u) for y in state_space) \
                  + np.dot(probs, V)

            V_new[x] = val

        V = V_new

    return V


def improve_policy_fh(V, state_space, action_spaces,
                      transition_probs, costs, horizon):
    """
    Greedy improvement for ALL stages
    """
    joint_actions = list(itertools.product(*action_spaces))
    m = len(action_spaces)

    policy = []

    for t in range(horizon):
        mu_t = [{x: None for x in state_space} for _ in range(m)]

        for x in state_space:
            best_val = np.inf
            best_u = None

            for u in joint_actions:
                probs = transition_probs(x, u)
                val = sum(probs[y] * costs(x, y, u) for y in state_space) \
                      + np.dot(probs, V)

                if val < best_val:
                    best_val = val
                    best_u = u

            for i in range(m):
                mu_t[i][x] = best_u[i]

        policy.append(mu_t)

    return policy


# def exact_fh_policy_iteration(transition_probs, costs,
#                              state_space, action_spaces,
#                              horizon, terminal_cost,
#                              n_iter=10):
#     """
#     Finite Horizon Policy Iteration + rollout tracking
#     """
#
#     # 🔹 initialize random policy
#     m = len(action_spaces)
#     policy = []
#     for t in range(horizon):
#         mu_t = [{x: np.random.choice(action_spaces[i]) for x in state_space}
#                 for i in range(m)]
#         policy.append(mu_t)
#
#     trajectory = []
#
#     for it in range(n_iter):
#
#         # 🔹 policy evaluation
#         V = evaluate_policy_fh(policy, transition_probs, costs,
#                                state_space, horizon, terminal_cost)
#
#         # 🔹 policy improvement
#         policy = improve_policy_fh(V, state_space, action_spaces,
#                                   transition_probs, costs, horizon)
#
#         # 🔥 rollout reward (THIS is what matters)
#         r = rollout_fh(policy, transition_probs, costs,
#                        state_space, horizon)
#         trajectory.append(r)
#
#     return policy, trajectory

def DPI_ALP_FH_iterative(transition_probs, costs, state_space, action_spaces,
                         Phi, c, horizon, terminal_cost, T_outer=30):
    m = len(action_spaces)
    mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    trajectory = []
    for t in range(T_outer):
        # Use current mu for evaluation
        policy = [mu for _ in range(horizon)]
        J = terminal_cost.copy()
        for k in reversed(range(horizon)):
            J = CACFN_FH(k, policy, J, transition_probs, costs, state_space, Phi, c)
        # Improve policy
        mu_updated = []
        for i in range(m):
            mu_base = [mu[j] for j in range(i+1, m)]
            mu_i_new = DPIm(i, mu_updated, mu_base, J, transition_probs, costs,
                            state_space, action_spaces, alpha=1.0)
            mu_updated.append(mu_i_new)
        mu = mu_updated
        # Rollout the NEW policy
        new_policy = [mu for _ in range(horizon)]
        r = rollout_fh(new_policy, transition_probs, costs, state_space, horizon, terminal_cost)
        trajectory.append(r)
    return trajectory


def exact_fh_policy_iteration(transition_probs, costs, state_space, action_spaces,
                              horizon, terminal_cost, gamma=1.0, T_outer=30):
    m = len(action_spaces)
    mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    trajectory = []
    for _ in range(T_outer):
        policy = [mu for _ in range(horizon)]
        # Policy evaluation
        V = terminal_cost.copy()
        for k in range(horizon-1, -1, -1):
            V_new = np.zeros(len(state_space))
            for x in state_space:
                u = tuple(policy[k][i][x] for i in range(m))
                probs = transition_probs(x, u)
                imm_cost = sum(probs[y] * costs(x, y, u) for y in state_space)
                future = np.dot(probs, V)
                V_new[x] = imm_cost + gamma * future
            V = V_new
        # Policy improvement
        mu = extract_greedy_policy(V, state_space, action_spaces, transition_probs, costs, gamma)
        # Rollout the NEW policy
        new_policy = [mu for _ in range(horizon)]
        r = rollout_fh(new_policy, transition_probs, costs, state_space, horizon, terminal_cost, gamma)
        trajectory.append(r)
    return trajectory

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def rollout_fh(policy, transition_probs, costs, state_space, horizon, terminal_cost, gamma=1.0):
    x = np.random.choice(state_space)
    total_cost = 0.0
    discount = 1.0
    
    for t in range(horizon):
        mu_t = policy[t]
        u = tuple(mu_t[i][x] for i in range(len(mu_t)))
        probs = transition_probs(x, u)
        y = np.random.choice(state_space, p=probs)
        cost = costs(x, y, u)
        total_cost += discount * cost
        discount *= gamma
        x = y
    total_cost += discount * terminal_cost[x]   # add terminal cost
    return -total_cost   # return reward

def rollout_ih(mu, transition_probs, costs, state_space, horizon=50, gamma=0.95):
    x = np.random.choice(state_space)
    total_reward = 0.0
    discount = 1.0

    for t in range(horizon):
        u = tuple(mu[i][x] for i in range(len(mu)))
        probs = transition_probs(x, u)
        y = np.random.choice(state_space, p=probs)

        cost = costs(x, y, u)
        reward = -cost

        total_reward += discount * reward
        discount *= gamma
        x = y

    return total_reward


def running_avg(arr):
    arr = np.array(arr)
    return np.cumsum(arr) / np.arange(1, len(arr)+1)


def extract_greedy_policy(V, state_space, action_spaces,
                         transition_probs, costs, gamma):
    """
    Extract optimal policy from value function V
    """
    m = len(action_spaces)
    joint_actions = list(itertools.product(*action_spaces))
    mu = [{x: None for x in state_space} for _ in range(m)]

    for x in state_space:
        best_val = np.inf
        best_u = None

        for u in joint_actions:
            probs = transition_probs(x, u)
            g = sum(probs[y] * costs(x, y, u) for y in state_space)
            val = g + gamma * np.dot(probs, V)

            if val < best_val:
                best_val = val
                best_u = u

        for i in range(m):
            mu[i][x] = best_u[i]

    return mu


def plot_fh(alp_fh, dp_fh, window=3, save_path=None):
    """ Plot rolling average for Finite Horizon: DPI-ALP vs Exact DP.
    Parameters
    ----------
    alp_fh, dp_fh : ndarray, shape (n_runs, n_iters)
    window : rolling-average window size
    save_path : file path to save figure (e.g. 'fh.png'). None = show.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw(ax,
          alp_fh, dp_fh,
          label_a='DPI-ALP FH (Algorithm 3)',
          label_b='Exact PI FH',
          col_a='#2563EB',
          col_b='#DC2626',
          title='Finite Horizon: DPI-ALP vs Exact PI FH\n' 
                f'(Average Total Reward, {alp_fh.shape[0]} runs)',
          window=window)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    else:
        plt.show()
        plt.close(fig)


def plot_ih(alp_ih, pi_ih, window=3, save_path=None):
    """ Plot rolling average for Infinite Horizon: DPI-ALP vs Exact PI.
    Parameters
    ----------
    alp_ih, pi_ih : ndarray, shape (n_runs, n_iters)
    window : rolling-average window size
    save_path : file path to save figure. None = show.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw(ax,
          alp_ih, pi_ih,
          label_a='DPI-ALP IH (Algorithm 5)',
          label_b='Exact PI IH',
          col_a='#16A34A',
          col_b='#D97706',
          title='Infinite Horizon: DPI-ALP vs Exact PI\n' 
                f'(Rolling Average Cost, {alp_ih.shape[0]} runs)',
          window=window)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    else:
        plt.show()
        plt.close(fig)


def rolling_mean(arr, window):
    """
    Compute rolling average along the last axis.
    Parameters
    ----------
    arr : ndarray, shape (n_runs, n_iters) or (n_iters,)
    window : int
    Returns
    -------
    ndarray, shape (n_runs, n_iters - window + 1) or (n_iters - window + 1,)
    """

    kernel = np.ones(window) / window
    if arr.ndim == 1:
        return np.convolve(arr, kernel, mode='valid')

    return np.array([np.convolve(row, kernel, mode='valid') for row in arr])

def _draw(ax, data_a, data_b, label_a, label_b, col_a, col_b, title, window):
    """ Draw rolling-mean lines + ±1 std shading for two algorithm arrays.
    Parameters
    ----------
    data_a, data_b : ndarray, shape (n_runs, n_iters)
    """

    # Align lengths — truncate to the shorter trajectory
    min_len = min(data_a.shape[1], data_b.shape[1])
    data_a = data_a[:, :min_len]
    data_b = data_b[:, :min_len]
    iters = np.arange(1, min_len + 1)
    ra = rolling_mean(data_a, window) #(n_runs, T')
    rb = rolling_mean(data_b, window)
    x = iters[window - 1:] # x-axis aligned to valid region

    mean_a, std_a = ra.mean(axis=0), ra.std(axis=0)
    mean_b, std_b = rb.mean(axis=0), rb.std(axis=0)

    # shaded ±1 std bands
    ax.fill_between(x, mean_a - std_a, mean_a + std_a, color=col_a, alpha=0.18)
    ax.fill_between(x, mean_b - std_b, mean_b + std_b, color=col_b, alpha=0.18)

    # mean lines
    ax.plot(x, mean_a, color=col_a, lw=2.2, label=label_a)
    ax.plot(x, mean_b, color=col_b, lw=2.2, linestyle='--', label=label_b)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Policy Iteration Step', fontsize=11)
    ax.set_ylabel(f'Mean Cost (rolling avg, window={window})', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)


if __name__ == "__main__":

    # --- Grid setup -----------------------------------------------------------
    GRID = 4          # 4x4 grid => 16 cells
    N_STATES_PER = GRID * GRID   # states per agent (position)
    ACTIONS = ["up", "down", "left", "right"]
    N_AGENTS = 2
    FLIES = [(0, 0), (3, 3)]     # fixed goal positions
    HORIZON = 10
    ALPHA = 0.95

    def idx(r, c):
        return r * GRID + c

    def pos(s):
        return divmod(s, GRID)

    def move(r, c, action):
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            return nr, nc
        return r, c  # boundary: stay, incur penalty elsewhere

    def out_of_grid(r, c, action):
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
        nr, nc = r + dr, c + dc
        return not (0 <= nr < GRID and 0 <= nc < GRID)

    # Joint states: pair of agent positions (stored as list of tuples for lookup)
    joint_states_list = list(itertools.product(range(N_STATES_PER), range(N_STATES_PER)))
    n_joint = len(joint_states_list)
    # state_space is always a list of INTEGER indices 0..n_joint-1
    state_space_int = list(range(n_joint))
    state_index = {s: i for i, s in enumerate(joint_states_list)}

    action_spaces = [list(range(len(ACTIONS))), list(range(len(ACTIONS)))]

    def transition_probs(x_int, u_joint):
        """Deterministic transitions. x_int is an integer index."""
        s0, s1 = joint_states_list[x_int]
        r0, c0 = pos(s0)
        r1, c1 = pos(s1)
        a0, a1 = ACTIONS[u_joint[0]], ACTIONS[u_joint[1]]
        nr0, nc0 = move(r0, c0, a0)
        nr1, nc1 = move(r1, c1, a1)
        ns = (idx(nr0, nc0), idx(nr1, nc1))
        probs = np.zeros(n_joint)
        probs[state_index[ns]] = 1.0
        return probs

    def env_costs(x_int, y_int, u_joint):
        """Single-stage cost. x_int, y_int are integer indices."""
        s0, s1 = joint_states_list[x_int]
        r0, c0 = pos(s0)
        r1, c1 = pos(s1)
        a0, a1 = ACTIONS[u_joint[0]], ACTIONS[u_joint[1]]
        cost = 1.0
        if out_of_grid(r0, c0, a0) or out_of_grid(r1, c1, a1):
            cost += 2.0
        nr0, nc0 = move(r0, c0, a0)
        nr1, nc1 = move(r1, c1, a1)
        if (nr0, nc0) == (nr1, nc1):
            cost += 2.0
        return cost

    # Terminal cost
    terminal_cost = np.ones(n_joint)

    # Feature matrix: RBF features over integer state indices
    d = 20

    # Build a constant feature plus RBF features (or just use one-hot for exact)
    Phi_constant = np.ones((n_joint, 1))
    # Keep original random RBFs
    np.random.seed(42)
    centres = np.random.choice(n_joint, d, replace=False)
    Phi_rbf = np.exp(-0.1 * np.abs(np.arange(n_joint)[:, None] - centres))
    Phi = np.hstack([Phi_constant, Phi_rbf])   # shape (n_joint, d+1)
    c_weights = np.ones(n_joint) / n_joint

    # print("=" * 60)
    # print("Demo: Finite Horizon DPI-ALP (Algorithm 3)")
    # print("=" * 60)
    # optimal_policy_fh, J_fh, traj_fh = DPI_ALP_FH(
    #     HORIZON, state_space_int, action_spaces,
    #     transition_probs, env_costs, terminal_cost,
    #     Phi, c_weights
    # )
    # print(f"Stage 0 approx. cost (first 5 states): {J_fh[0][:5].round(3)}")

    # print()
    # print("=" * 60)
    # print("Demo: Infinite Horizon DPI-ALP (Algorithm 5)")
    # print("=" * 60)
    # mu_ih, J_ih, traj_ih = DPI_ALP_IH(
    #     state_space_int, action_spaces,
    #     transition_probs, env_costs,
    #     Phi, c_weights, ALPHA, T=50
    # )
    # print(f"IH approx. cost (first 5 states): {J_ih[:5].round(3)}")

    traj_fh = DPI_ALP_FH_iterative(transition_probs, env_costs, state_space_int, action_spaces, Phi, c_weights, HORIZON, terminal_cost)

    # Exact finite horizon (backward induction)
    traj_exact_fh = exact_fh_policy_iteration(
        transition_probs,
        env_costs,
        state_space_int,
        action_spaces,
        horizon=HORIZON,
        terminal_cost=terminal_cost
    )
    # print("Exact FH (N=10):", V_exact_fh[:5].round(3))

    # Exact infinite horizon (value iteration with progress)
    # V_exact_ih, traj_exact_ih = exact_infinite_horizon(
    #     transition_probs, env_costs, state_space_int, action_spaces,
    #     gamma=ALPHA, tol=1e-8, max_iter=10000, verbose=True
    # )
    # print("Exact IH (γ=0.95):", V_exact_ih[:5].round(3))

    # traj_ih_avg = running_avg(traj_ih)
    # traj_exact_ih_avg = running_avg(traj_exact_ih)

    plot_fh(np.array([traj_fh]), np.array([traj_exact_fh]), window=3, save_path='rolling_avg_fh.png')
    # plot_ih(np.array([traj_ih_avg]), np.array([traj_exact_ih_avg]), window=3, save_path='rolling_avg_ih.png')
    print("Done.")

