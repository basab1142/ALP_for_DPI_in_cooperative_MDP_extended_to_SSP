"""
Reproduce Figure 2 (SSP version) from the paper:
"Approximate Linear Programming for Decentralized Policy Iteration
 in Cooperative Multi-Agent Markov Decision Processes"
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ========================== Environment (Flies‑Spiders with absorbing flies) ==========================
GRID = 4
N_STATES_PER = GRID * GRID
ACTIONS = ["up", "down", "left", "right"]
N_AGENTS = 2
HORIZON = 10          # not used in SSP, but kept for compatibility
ALPHA = 0.95          # not used in SSP

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

# Identify absorbing states (both flies caught)
absorbing = [i for i, (_, _, f0, f1) in enumerate(extended_states) if f0 and f1]
non_absorbing = [i for i in state_space if i not in absorbing]

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
        ns = x_int
    else:
        ns = state_to_idx[(pos_to_idx(nr0, nc0), pos_to_idx(nr1, nc1), nf0, nf1)]

    probs = np.zeros(n_states)
    probs[ns] = 1.0
    return probs

def env_costs(x_int, y_int, u_joint):
    s0, s1, f0, f1 = extended_states[x_int]
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

# Features: one‑hot (exact representation)
Phi = np.eye(n_states)
c_weights = np.ones(n_states) / n_states


# ========================== Proper initial policy ==========================
def get_proper_initial_policy(state_space, action_spaces):
    """Agent0 moves toward fly at (0,0), Agent1 toward fly at (3,3)."""
    m = len(action_spaces)
    mu = [{x: None for x in state_space} for _ in range(m)]
    fly0 = (0, 0)
    fly1 = (3, 3)
    for x in state_space:
        s0, s1, f0, f1 = extended_states[x]
        # If fly already caught, stay put (action 0 = up, but any action works)
        if f0:
            mu[0][x] = 0
        else:
            r0, c0 = pos(s0)
            # Simple Manhattan direction
            dr = np.sign(fly0[0] - r0)
            dc = np.sign(fly0[1] - c0)
            if dr == -1: action = 1  # down
            elif dr == 1: action = 0 # up
            elif dc == -1: action = 3 # right? Actually need mapping: up=0, down=1, left=2, right=3
            else: action = 0
            # Better: map direction to action index
            # We'll use a simple rule: if row diff > 0 move down (1), else if row diff < 0 move up (0)
            # else if col diff > 0 move right (3), else left (2)
            if dr > 0: action = 1
            elif dr < 0: action = 0
            elif dc > 0: action = 3
            elif dc < 0: action = 2
            else: action = 0
            mu[0][x] = action
        # Same for agent1
        if f1:
            mu[1][x] = 0
        else:
            r1, c1 = pos(s1)
            dr = np.sign(fly1[0] - r1)
            dc = np.sign(fly1[1] - c1)
            if dr > 0: action = 1
            elif dr < 0: action = 0
            elif dc > 0: action = 3
            elif dc < 0: action = 2
            else: action = 0
            mu[1][x] = action
    return mu

# ========================== Core algorithms ==========================
def DPIm(i, mu_updated, mu_base, J, trans, costs, state_space, action_spaces, alpha=1.0):
    """Decentralized policy improvement (same as before, alpha=1 for SSP)."""
    m = len(mu_updated) + 1 + len(mu_base)
    mu_i_new = {}
    for x in state_space:
        best_val = np.inf
        best_a = None
        for ui in action_spaces[i]:
            u_joint = []
            for k in range(i):
                u_joint.append(mu_updated[k][x])
            u_joint.append(ui)
            for k in range(i+1, m):
                u_joint.append(mu_base[k-i-1][x])
            u_joint = tuple(u_joint)
            probs = trans(x, u_joint)
            g_vals = np.array([costs(x, y, u_joint) for y in state_space])
            val = np.dot(probs, g_vals + alpha * J)
            if val < best_val:
                best_val = val
                best_a = ui
        mu_i_new[x] = best_a
    return mu_i_new

def CACFN_SSP(mu, trans, costs, state_space, Phi, c):
    """ALP for SSP: solve max c^T V s.t. V <= T_mu V, with V(absorbing)=0."""
    n, d = Phi.shape
    m_agents = len(mu)

    # Build transition and cost for the given policy
    P_mu = np.zeros((n, n))
    g_mu = np.zeros(n)
    for idx, x in enumerate(state_space):
        u = tuple(mu[i][x] for i in range(m_agents))
        probs = trans(x, u)
        P_mu[idx] = probs
        g_mu[idx] = np.dot(probs, [costs(x, y, u) for y in state_space])

    # Constraint matrix: (I - P_mu) @ V <= g_mu
    A_ub = (np.eye(n) - P_mu) @ Phi
    b_ub = g_mu

    # Equality constraints: V(x) = 0 for absorbing states
    A_eq = []
    b_eq = []
    for idx in absorbing:
        A_eq.append(Phi[idx])
        b_eq.append(0.0)

    c_lp = -(Phi.T @ c)   # minimize -c^T V
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(None, None)]*d, method='highs')
    if res.success:
        r_opt = res.x
    else:
        r_opt = np.zeros(d)
    return Phi @ r_opt


def rollout_ssp(mu, trans, costs, state_space, max_steps=500, n_episodes=100):
    total_cost = 0.0
    for _ in range(n_episodes):
        x = np.random.choice(state_space)
        cost = 0.0
        for step in range(max_steps):
            s0, s1, f0, f1 = extended_states[x]
            if f0 and f1:
                break
            u = tuple(mu[i][x] for i in range(len(mu)))
            probs = trans(x, u)
            y = np.random.choice(state_space, p=probs)
            cost += costs(x, y, u)
            x = y
        else:
            # Not absorbed within max_steps → large penalty
            cost += 100.0
        total_cost += cost
    return -total_cost / n_episodes


def extract_greedy_policy(V, state_space, action_spaces, trans, costs, gamma=1.0):
    """Greedy policy w.r.t. value function V (gamma=1 for SSP)."""
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


# ========================== DPI-ALP for SSP ==========================
def DPI_ALP_SSP_iterative(trans, costs, state_space, action_spaces, Phi, c, T_outer=30):
    m = len(action_spaces)
    mu = get_proper_initial_policy(state_space, action_spaces)
    traj = []
    prev_mu = None
    for t in range(T_outer):
        V = CACFN_SSP(mu, trans, costs, state_space, Phi, c)
        mu_updated = []
        for i in range(m):
            mu_base = [mu[j] for j in range(i+1, m)]
            mu_i = DPIm(i, mu_updated, mu_base, V, trans, costs, state_space, action_spaces, alpha=1.0)
            mu_updated.append(mu_i)
        mu_new = mu_updated
        r = rollout_ssp(mu_new, trans, costs, state_space)
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
                print(f"  SSP ALP converged at iteration {t+1}")
                break
        prev_mu = mu_new
        mu = mu_new
    return traj


# ========================== Exact SSP Policy Iteration ==========================
def exact_ssp_pi(trans, costs, state_space, action_spaces, T_outer=30):
    m = len(action_spaces)
    joint_actions = list(itertools.product(*action_spaces))
    absorbing = [i for i, (_, _, f0, f1) in enumerate(extended_states) if f0 and f1]
    non_absorbing = [i for i in state_space if i not in absorbing]
    mu = get_proper_initial_policy(state_space, action_spaces)  # proper initial
    traj = []

    for _ in range(T_outer):
        n = len(state_space)
        P_mu = np.zeros((n, n))
        g_mu = np.zeros(n)
        for idx, x in enumerate(state_space):
            u = tuple(mu[i][x] for i in range(m))
            probs = trans(x, u)
            P_mu[idx] = probs
            g_mu[idx] = np.dot(probs, [costs(x, y, u) for y in state_space])
        n_na = len(non_absorbing)
        I_na = np.eye(n_na)
        P_na = P_mu[np.ix_(non_absorbing, non_absorbing)]
        g_na = g_mu[non_absorbing]
        # Regularization to handle improper policies
        reg = 1e-6
        V_na = np.linalg.solve(I_na - P_na + reg * np.eye(n_na), g_na)
        V = np.zeros(n)
        V[non_absorbing] = V_na
        mu = extract_greedy_policy(V, state_space, action_spaces, trans, costs, gamma=1.0)
        traj.append(rollout_ssp(mu, trans, costs, state_space))
    return traj


# ========================== Run multiple seeds and plot ==========================
def run_experiments(n_seeds=10):
    alp_trajs = []
    exact_trajs = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        traj_alp = DPI_ALP_SSP_iterative(transition_probs, env_costs, state_space,
                                         action_spaces, Phi, c_weights, T_outer=30)
        traj_exact = exact_ssp_pi(transition_probs, env_costs, state_space,
                                  action_spaces, T_outer=30)
        alp_trajs.append(traj_alp)
        exact_trajs.append(traj_exact)
    return alp_trajs, exact_trajs

def pad_trajectories(traj_list):
    max_len = max(len(t) for t in traj_list)
    padded = []
    for t in traj_list:
        if len(t) < max_len:
            t_padded = t + [t[-1]] * (max_len - len(t))
        else:
            t_padded = t
        padded.append(t_padded)
    return padded


def plot_ssp(alp_trajs, exact_trajs, window=3, save_path='ssp.png'):
    alp_arr = np.array(pad_trajectories(alp_trajs))
    exact_arr = np.array(pad_trajectories(exact_trajs))
    # Align lengths
    min_len = min(alp_arr.shape[1], exact_arr.shape[1])
    alp_arr = alp_arr[:, :min_len]
    exact_arr = exact_arr[:, :min_len]
    iters = np.arange(1, min_len + 1)

    def rolling_mean(data, w):
        return np.array([np.convolve(row, np.ones(w) / w, mode='valid') for row in data])

    alp_roll = rolling_mean(alp_arr, window)
    exact_roll = rolling_mean(exact_arr, window)
    x = iters[window - 1:]

    mean_alp, std_alp = alp_roll.mean(axis=0), alp_roll.std(axis=0)
    mean_exact, std_exact = exact_roll.mean(axis=0), exact_roll.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Shaded region (semi‑transparent)
    ax.fill_between(x, mean_alp - std_alp, mean_alp + std_alp, alpha=0.2, color='green')
    ax.fill_between(x, mean_exact - std_exact, mean_exact + std_exact, alpha=0.2, color='orange')
    # Solid lines with markers
    ax.plot(x, mean_alp, color='green', linewidth=2, marker='o', markersize=4, label='DPI-ALP (SSP)')
    ax.plot(x, mean_exact, color='orange', linewidth=2, marker='s', markersize=4, linestyle='--',
            label='Exact Policy Iteration (SSP)')

    ax.set_xlabel('Policy Iteration Step')
    ax.set_ylabel('Average Total Reward')
    ax.set_title('Stochastic Shortest Path (undiscounted)')
    ax.legend()
    ax.grid(True, linestyle=':')

    # Ensure y-axis includes the data range with some margin
    y_min = min(mean_alp.min(), mean_exact.min()) - 0.5
    y_max = max(mean_alp.max(), mean_exact.max()) + 0.5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print("Running SSP experiments with 10 random seeds...")
    alp, exact = run_experiments(n_seeds=10)
    print(alp[0])

    plot_ssp(alp, exact, window=3, save_path='ssp.png')
    print("Done. Figure saved as ssp.png")