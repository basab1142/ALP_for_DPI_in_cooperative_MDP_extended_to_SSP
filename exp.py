"""
Reproduce Figure 2 (for App.1) from the paper:
"Approximate Linear Programming for Decentralized Policy Iteration
 in Cooperative Multi-Agent Markov Decision Processes"
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ========================== Environment (same as before) ==========================
# ========================== Environment (Flies-Spiders with absorbing flies) ==========================
# ========================== Environment (Flies-Spiders with absorbing flies) ==========================
GRID = 4
N_STATES_PER = GRID * GRID
ACTIONS = ["up", "down", "left", "right"]
N_AGENTS = 2
HORIZON = 10
ALPHA = 0.95


def pos_to_idx(r, c):
    return r * GRID + c


def pos(s):
    return divmod(s, GRID)


def move(r, c, action):
    dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID and 0 <= nc < GRID:
        return nr, nc
    return r, c


def out_of_grid(r, c, action):
    dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
    nr, nc = r + dr, c + dc
    return not (0 <= nr < GRID and 0 <= nc < GRID)


fly_positions = {(0, 0), (3, 3)}

# Build extended state space: (pos0, pos1, f0_caught, f1_caught)
extended_states = []
state_to_idx = {}
state_idx = 0
for s0 in range(N_STATES_PER):
    for s1 in range(N_STATES_PER):
        for f0 in (0, 1):
            for f1 in (0, 1):
                extended_states.append((s0, s1, f0, f1))
                state_to_idx[(s0, s1, f0, f1)] = state_idx
                state_idx += 1
n_states = len(extended_states)
state_space = list(range(n_states))


def get_positions_and_flags(state_idx):
    return extended_states[state_idx]


action_spaces = [list(range(len(ACTIONS))), list(range(len(ACTIONS)))]


def transition_probs(x_int, u_joint):
    s0, s1, f0, f1 = extended_states[x_int]
    a0, a1 = ACTIONS[u_joint[0]], ACTIONS[u_joint[1]]
    r0, c0 = pos(s0)
    r1, c1 = pos(s1)

    if f0:
        nr0, nc0 = r0, c0
    else:
        nr0, nc0 = move(r0, c0, a0)
    if f1:
        nr1, nc1 = r1, c1
    else:
        nr1, nc1 = move(r1, c1, a1)

    nf0 = f0
    nf1 = f1
    if not f0 and (nr0, nc0) in fly_positions:
        nf0 = 1
    if not f1 and (nr1, nc1) in fly_positions:
        nf1 = 1

    # If both caught, stay in the same state (absorbing)
    if f0 and f1:
        ns = x_int  # stay in same state (already absorbing)
    else:
        ns = state_to_idx[(pos_to_idx(nr0, nc0), pos_to_idx(nr1, nc1), nf0, nf1)]

    probs = np.zeros(n_states)
    probs[ns] = 1.0
    return probs


def env_costs(x_int, y_int, u_joint):
    s0, s1, f0, f1 = extended_states[x_int]
    # If both flies already caught, cost = 0
    if f0 and f1:
        return 0.0
    a0, a1 = ACTIONS[u_joint[0]], ACTIONS[u_joint[1]]
    r0, c0 = pos(s0)
    r1, c1 = pos(s1)
    cost = 1.0
    if not f0 and out_of_grid(r0, c0, a0):
        cost += 2.0
    if not f1 and out_of_grid(r1, c1, a1):
        cost += 2.0
    if f0:
        nr0, nc0 = r0, c0
    else:
        nr0, nc0 = move(r0, c0, a0)
    if f1:
        nr1, nc1 = r1, c1
    else:
        nr1, nc1 = move(r1, c1, a1)
    if (nr0, nc0) == (nr1, nc1):
        cost += 2.0
    return cost


# Terminal cost for finite horizon: zero if both flies caught, else penalty
terminal_cost = np.zeros(n_states)
for i, (s0, s1, f0, f1) in enumerate(extended_states):
    if f0 and f1:
        terminal_cost[i] = 0.0
    else:
        terminal_cost[i] = 10.0  # penalty for not catching both flies

# Features: one-hot (exact representation)
Phi = np.eye(n_states)
# np.random.seed(42)
# centres = np.random.choice(n_joint, 20, replace=False)
# Phi_rbf = np.exp(-0.1 * np.abs(np.arange(n_joint)[:, None] - centres))
# Phi = np.hstack([Phi_const, Phi_rbf])
c_weights = np.ones(n_states) / n_states


# ========================== Core algorithms (from earlier) ==========================
def DPIm(i, mu_updated, mu_base, J, trans, costs, state_space, action_spaces, alpha=1.0):
    m = len(mu_updated) + 1 + len(mu_base)
    mu_i_new = {}
    for x in state_space:
        best_val = np.inf
        best_a = None
        for ui in action_spaces[i]:
            u_joint = []
            for k in range(i): u_joint.append(mu_updated[k][x])
            u_joint.append(ui)
            for k in range(i+1, m): u_joint.append(mu_base[k-i-1][x])
            u_joint = tuple(u_joint)
            probs = trans(x, u_joint)
            g_vals = np.array([costs(x, y, u_joint) for y in state_space])
            val = np.dot(probs, g_vals + alpha * J)
            if val < best_val:
                best_val = val
                best_a = ui
        mu_i_new[x] = best_a
    return mu_i_new

def CACFN_FH(k, pi, J_next, trans, costs, state_space, Phi, c):
    n, d = Phi.shape
    mu_k = pi[k]
    b = np.zeros(n)
    for idx, x in enumerate(state_space):
        u = tuple(mu_k[i][x] for i in range(len(mu_k)))
        probs = trans(x, u)
        g_vals = np.array([costs(x, y, u) for y in state_space])
        b[idx] = np.dot(probs, g_vals + J_next)
    c_lp = -Phi.T @ c
    res = linprog(c_lp, A_ub=Phi, b_ub=b, bounds=[(None,None)]*d, method='highs')
    if res.success:
        r_opt = res.x
    else:
        r_opt = np.zeros(d)
    return Phi @ r_opt

def CACFN_IH(mu, trans, costs, state_space, Phi, c, alpha):
    n, d = Phi.shape
    m = len(mu)
    P_mu = np.zeros((n,n))
    g_mu = np.zeros(n)
    for idx, x in enumerate(state_space):
        u = tuple(mu[i][x] for i in range(m))
        probs = trans(x, u)
        P_mu[idx] = probs
        g_mu[idx] = np.dot(probs, [costs(x, y, u) for y in state_space])
    A = (np.eye(n) - alpha * P_mu) @ Phi
    b = g_mu
    c_lp = -(Phi.T @ c)
    res = linprog(c_lp, A_ub=A, b_ub=b, bounds=[(None,None)]*d, method='highs')
    if res.success:
        r_opt = res.x
    else:
        r_opt = np.zeros(d)
    return Phi @ r_opt

def rollout_fh(policy, trans, costs, state_space, horizon, term_cost, gamma=1.0, n_episodes=100):
    total_reward = 0.0
    for _ in range(n_episodes):
        x = np.random.choice(state_space)
        total_cost = 0.0
        disc = 1.0
        for t in range(horizon):
            u = tuple(policy[t][i][x] for i in range(len(policy[0])))
            probs = trans(x, u)
            y = np.random.choice(state_space, p=probs)
            total_cost += disc * costs(x, y, u)
            disc *= gamma
            x = y
        total_cost += disc * term_cost[x]
        total_reward += -total_cost
    return total_reward / n_episodes


def rollout_ih(mu, trans, costs, state_space, horizon=500, gamma=0.95, n_episodes=100):
    n = len(state_space)
    m = len(mu)
    P = np.zeros((n, n))
    g = np.zeros(n)

    for idx, x in enumerate(state_space):
        u = tuple(mu[i][x] for i in range(m))
        probs = trans(x, u)
        P[idx] = probs
        g[idx] = sum(probs[y] * costs(x, y, u) for y in state_space)
    V = np.linalg.solve(np.eye(n) - gamma * P, g)

    return -np.mean(V)  # negative cost = reward


def extract_greedy_policy(V, state_space, action_spaces, trans, costs, gamma):
    m = len(action_spaces)
    joint_actions = list(itertools.product(*action_spaces))
    mu = [{x: None for x in state_space} for _ in range(m)]
    for x in state_space:
        best_val = np.inf
        best_u = None
        for u in joint_actions:
            probs = trans(x, u)
            g = sum(probs[y] * costs(x, y, u) for y in state_space)
            val = g + gamma * np.dot(probs, V)
            if val < best_val - 1e-9:
                best_val = val
                best_u = u
        for i, a in enumerate(best_u):
            mu[i][x] = a
    return mu

# ========================== DPI-ALP Iterative (Finite) ==========================
def DPI_ALP_FH_iterative(trans, costs, state_space, action_spaces, Phi, c,
                         horizon, term_cost, T_outer=30):
    m = len(action_spaces)
    mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    traj = []
    prev_mu = None
    for _ in range(T_outer):
        policy = [mu for _ in range(horizon)]
        J = term_cost.copy()
        for k in reversed(range(horizon)):
            J = CACFN_FH(k, policy, J, trans, costs, state_space, Phi, c)
        mu_updated = []
        for i in range(m):
            mu_base = [mu[j] for j in range(i+1, m)]
            mu_i = DPIm(i, mu_updated, mu_base, J, trans, costs, state_space, action_spaces, alpha=1.0)
            mu_updated.append(mu_i)
        mu_new = mu_updated
        new_policy = [mu_new for _ in range(horizon)]
        r = rollout_fh(new_policy, trans, costs, state_space, horizon, term_cost)
        traj.append(r)

        # Policy convergence check
        if prev_mu is not None:
            converged = True
            for i in range(m):
                for x in state_space:
                    if mu_new[i][x] != prev_mu[i][x]:
                        converged = False
                        break
                if not converged:
                    break
            if converged:
                print(f"  FH converged at iteration {len(traj)} (policy stable)")
                break
        prev_mu = mu_new
        mu = mu_new
    return traj

# ========================== Exact Finite Horizon DP (backward) ==========================
def exact_fh_dp(trans, costs, state_space, action_spaces, horizon, term_cost, gamma=1.0):
    joint_actions = list(itertools.product(*action_spaces))
    V = term_cost.copy()
    # Build optimal policy (stationary) – we'll do one backward pass
    # But for a fair comparison, we need a policy that we can rollout.
    # We'll compute the optimal value and then the greedy policy from stage 0.
    for _ in range(horizon):
        V_new = np.zeros(len(state_space))
        for x in state_space:
            best = np.inf
            for u in joint_actions:
                probs = trans(x, u)
                imm = sum(probs[y] * costs(x, y, u) for y in state_space)
                fut = np.dot(probs, V)
                val = imm + gamma * fut
                if val < best:
                    best = val
            V_new[x] = best
        V = V_new
    # Greedy policy from V (stage 0)
    mu = extract_greedy_policy(V, state_space, action_spaces, trans, costs, gamma)
    policy = [mu for _ in range(horizon)]
    # Rollout gives optimal total reward
    return rollout_fh(policy, trans, costs, state_space, horizon, term_cost, gamma)

# ========================== DPI-ALP Iterative (Infinite) ==========================
def DPI_ALP_IH_iterative(trans, costs, state_space, action_spaces, Phi, c,
                         alpha, T_outer=30, eval_steps=50):
    m = len(action_spaces)
    mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    traj = []
    prev_mu = None
    for t in range(T_outer):
        J = CACFN_IH(mu, trans, costs, state_space, Phi, c, alpha)
        mu_updated = []
        for i in range(m):
            mu_base = [mu[j] for j in range(i+1, m)]
            mu_i = DPIm(i, mu_updated, mu_base, J, trans, costs, state_space, action_spaces, alpha)
            mu_updated.append(mu_i)
        mu_new = mu_updated
        r = rollout_ih(mu_new, trans, costs, state_space, eval_steps, alpha)
        traj.append(r)
        # Policy convergence check (explicit loop)
        if prev_mu is not None:
            converged = True
            for i in range(m):
                for x in state_space:
                    if mu_new[i][x] != prev_mu[i][x]:
                        converged = False
                        break
                if not converged:
                    break
            if converged:
                print(f"  Converged at iteration {t+1} (policy stable)")
                break
        prev_mu = mu_new
        mu = mu_new
    return traj


# ========================== Exact Infinite Horizon Policy Iteration ==========================
def exact_ih_pi(trans, costs, state_space, action_spaces, gamma, T_outer=30, eval_steps=50):
    m = len(action_spaces)
    joint_actions = list(itertools.product(*action_spaces))
    # random initial policy
    mu = [{x: np.random.choice(action_spaces[i]) for x in state_space} for i in range(m)]
    traj = []
    for _ in range(T_outer):
        # Policy evaluation: solve linear system (I - gamma P_mu) V = g_mu
        n = len(state_space)
        P_mu = np.zeros((n, n))
        g_mu = np.zeros(n)
        for idx, x in enumerate(state_space):
            u = tuple(mu[i][x] for i in range(m))
            probs = trans(x, u)
            P_mu[idx] = probs
            g_mu[idx] = sum(probs[y] * costs(x, y, u) for y in state_space)
        V = np.linalg.solve(np.eye(n) - gamma * P_mu, g_mu)
        # Policy improvement
        mu = extract_greedy_policy(V, state_space, action_spaces, trans, costs, gamma)
        # Rollout
        r = rollout_ih(mu, trans, costs, state_space, eval_steps, gamma)
        traj.append(r)
    return traj

# ========================== Run multiple seeds and plot ==========================
def run_experiments(n_seeds=10):
    fh_alp_all = []
    fh_exact_all = []
    ih_alp_all = []
    ih_exact_all = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        # Finite horizon
        traj_fh = DPI_ALP_FH_iterative(transition_probs, env_costs, state_space, action_spaces,
                                       Phi, c_weights, HORIZON, terminal_cost, T_outer=30)
        fh_alp_all.append(traj_fh)
        exact_val = exact_fh_dp(transition_probs, env_costs, state_space, action_spaces,
                                HORIZON, terminal_cost)
        fh_exact_all.append([exact_val] * len(traj_fh))
        # Infinite horizon
        traj_ih = DPI_ALP_IH_iterative(transition_probs, env_costs, state_space, action_spaces,
                                       Phi, c_weights, ALPHA, T_outer=30, eval_steps=50)
        ih_alp_all.append(traj_ih)
        traj_ih_exact = exact_ih_pi(transition_probs, env_costs, state_space, action_spaces,
                                    ALPHA, T_outer=30, eval_steps=50)
        ih_exact_all.append(traj_ih_exact)

    def pad(traj_list):
        max_len = max(len(t) for t in traj_list)
        padded = []
        for t in traj_list:
            if len(t) < max_len:
                t_padded = t + [t[-1]] * (max_len - len(t))
            else:
                t_padded = t
            padded.append(t_padded)
        return padded

    fh_alp_all = pad(fh_alp_all)
    fh_exact_all = pad(fh_exact_all)
    ih_alp_all = pad(ih_alp_all)
    ih_exact_all = pad(ih_exact_all)

    return fh_alp_all, fh_exact_all, ih_alp_all, ih_exact_all


def plot_results(fh_alp, fh_exact, ih_alp, ih_exact, window=3):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Finite horizon
    alp_arr = np.array(fh_alp)
    exact_arr = np.array(fh_exact)
    min_len = min(alp_arr.shape[1], exact_arr.shape[1])
    alp_arr = alp_arr[:, :min_len]
    exact_arr = exact_arr[:, :min_len]
    iters = np.arange(1, min_len+1)

    # rolling average
    def rolling_mean(data, w):
        return np.array([np.convolve(row, np.ones(w)/w, mode='valid') for row in data])
    alp_roll = rolling_mean(alp_arr, window)
    exact_roll = rolling_mean(exact_arr, window)
    x = iters[window-1:]

    mean_alp, std_alp = alp_roll.mean(axis=0), alp_roll.std(axis=0)
    mean_exact, std_exact = exact_roll.mean(axis=0), exact_roll.std(axis=0)

    ax = axes[0]
    ax.fill_between(x, mean_alp - std_alp, mean_alp + std_alp, alpha=0.3, color='blue')
    ax.plot(x, mean_alp, color='blue', label='DPI-ALP (Algorithm 3)')
    ax.fill_between(x, mean_exact - std_exact, mean_exact + std_exact, alpha=0.3, color='red')
    ax.plot(x, mean_exact, '--', color='red', label='Exact DP')
    ax.set_xlabel('Policy Iteration Step')
    ax.set_ylabel('Average Total Reward')
    ax.set_title(f'Finite Horizon (N={HORIZON})')
    ax.legend()
    ax.grid(True, linestyle=':')

    # Infinite horizon
    alp_arr_i = np.array(ih_alp)
    exact_arr_i = np.array(ih_exact)
    min_len_i = min(alp_arr_i.shape[1], exact_arr_i.shape[1])
    alp_arr_i = alp_arr_i[:, :min_len_i]
    exact_arr_i = exact_arr_i[:, :min_len_i]
    iters_i = np.arange(1, min_len_i+1)
    alp_roll_i = rolling_mean(alp_arr_i, window)
    exact_roll_i = rolling_mean(exact_arr_i, window)
    x_i = iters_i[window-1:]

    mean_alp_i, std_alp_i = alp_roll_i.mean(axis=0), alp_roll_i.std(axis=0)
    mean_exact_i, std_exact_i = exact_roll_i.mean(axis=0), exact_roll_i.std(axis=0)

    ax = axes[1]
    ax.fill_between(x_i, mean_alp_i - std_alp_i, mean_alp_i + std_alp_i, alpha=0.3, color='green')
    ax.plot(x_i, mean_alp_i, color='green', label='DPI-ALP (Algorithm 5)')
    ax.fill_between(x_i, mean_exact_i - std_exact_i, mean_exact_i + std_exact_i, alpha=0.3, color='orange')
    ax.plot(x_i, mean_exact_i, '--', color='orange', label='Regular Policy Iteration')
    ax.set_xlabel('Policy Iteration Step')
    ax.set_ylabel('Average Total Reward')
    ax.set_title(f'Infinite Horizon (γ={ALPHA})')
    ax.legend()
    ax.grid(True, linestyle=':')

    plt.tight_layout()
    plt.savefig('fig2_app1.png', dpi=150)


if __name__ == "__main__":
    print("Running experiments with 10 random seeds...")
    fh_alp, fh_exact, ih_alp, ih_exact = run_experiments(n_seeds=10)
    plot_results(fh_alp, fh_exact, ih_alp, ih_exact, window=3)
    print("Done. Figure saved as fig2_app1.png")