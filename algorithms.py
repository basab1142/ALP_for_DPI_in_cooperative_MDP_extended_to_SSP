import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm


def approximate_linear_programming(env, max_iterations=100):

    agents = env.num_agents
    grid = env.grid_size

    # 🔥 FIX 1: remove constant feature
    feature_dimension = agents + 2

    num_states = grid ** (2 * agents)

    # policy init
    policy = [np.random.randint(0, env.actions, size=(grid, grid, grid, grid)) for _ in range(agents)]

    # Φ matrix
    Phi = np.zeros((num_states, feature_dimension))
    d = Phi.shape[1]

    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                for l in range(grid):

                    idx = i*grid**3 + j*grid**2 + k*grid + l

                    max_dist = 2 * grid

                    d1 = abs(i - env.fly_pos[0]) + abs(j - env.fly_pos[1])
                    d2 = abs(k - env.fly_pos[0]) + abs(l - env.fly_pos[1])

                    interaction = abs(i - k) + abs(j - l)
                    d1 = d1 / max_dist
                    d2 = d2 / max_dist
                    interaction = interaction / max_dist
                    proximity = 0.5 / (1 + interaction * max_dist)

                    Phi[idx, 0] = d1
                    Phi[idx, 1] = d2
                    Phi[idx, 2] = interaction
                    Phi[idx, 3] = proximity

    def get_idx(s):
        return s[0]*grid**3 + s[1]*grid**2 + s[2]*grid + s[3]

    # objective
    c = np.ones(num_states) / num_states
    epsilon_weight = 10000
    weights = np.zeros(feature_dimension)

    reg = 0.01
    objective = np.append(-(Phi.T @ c).flatten() + reg * np.ones(d), epsilon_weight)

    for iteration in tqdm(range(max_iterations), desc="ALP Iterations"):

        old_policy = [p.copy() for p in policy]

        A_ub, b_ub, A_eq, b_eq = build_alp(env, policy, Phi)

        # 🔥 FIX: bounds prevent unboundedness
        bounds = [(-100, 100)] * 4 + [(0, None)]

        res = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            print("LP failed:", res.message)
            continue

        weights = res.x.copy()
        r = weights[:-1]
        # enforce symmetry
        r[0], r[1] = (r[0] + r[1]) / 2, (r[0] + r[1]) / 2

        # 🔁 DPIm
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    for l in range(grid):

                        s = np.array([i,j,k,l])

                        # agent 1
                        best_q = float("inf")

                        for a1 in range(env.actions):

                            a2_fixed = policy[1][i, j, k, l]

                            fs, c1, c2 = env.simulate_joint_step(s, a1, a2_fixed)

                            is_cap = (
                                    (fs[0] == env.fly_pos[0] and fs[1] == env.fly_pos[1]) or
                                    (fs[2] == env.fly_pos[0] and fs[3] == env.fly_pos[1])
                            )

                            cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))

                            q = cost + np.dot(Phi[get_idx(fs)], r)

                            if q < best_q:
                                best_q = q
                                policy[0][i, j, k, l] = a1

                        # agent 2
                        best_q = float("inf")

                        for a2 in range(env.actions):

                            a1_fixed = policy[0][i, j, k, l]

                            fs, c1, c2 = env.simulate_joint_step(s, a1_fixed, a2)

                            is_cap = (
                                    (fs[0] == env.fly_pos[0] and fs[1] == env.fly_pos[1]) or
                                    (fs[2] == env.fly_pos[0] and fs[3] == env.fly_pos[1])
                            )

                            cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))

                            q = cost + np.dot(Phi[get_idx(fs)], r)

                            if q < best_q:
                                best_q = q
                                policy[1][i, j, k, l] = a2

        # 🔥 FIX: convergence check
        if all(np.array_equal(policy[i], old_policy[i]) for i in range(agents)):
            print("Converged.")
            break

    return policy, weights


def build_alp(env, policy, Phi):

    d = Phi.shape[1]
    grid = env.grid_size

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    def get_idx(s):
        return s[0]*grid**3 + s[1]*grid**2 + s[2]*grid + s[3]

    # 🔥 FIX: sampling
    all_states = [(i,j,k,l) for i in range(grid)
                              for j in range(grid)
                              for k in range(grid)
                              for l in range(grid)]

    sampled = np.random.choice(len(all_states), 80, replace=False)

    for idx_s in sampled:
        i,j,k,l = all_states[idx_s]
        s = np.array([i,j,k,l])
        s_idx = get_idx(s)

        is_terminal = (
            (i==env.fly_pos[0] and j==env.fly_pos[1]) or
            (k==env.fly_pos[0] and l==env.fly_pos[1])
        )

        # 🔥 FIX: terminal as inequality
        if is_terminal:
            row = np.zeros(d+1)
            row[:d] = Phi[s_idx]
            A_ub.append(row)
            b_ub.append(0)
            continue

        a1 = policy[0][i,j,k,l]
        a2 = policy[1][i,j,k,l]

        fs, c1, c2 = env.simulate_joint_step(s, a1, a2)

        lhs = np.zeros(d+1)
        lhs[:d] = Phi[s_idx]
        lhs[-1] = -1.0

        ns_idx = get_idx(fs)

        is_cap = (
            (fs[0]==env.fly_pos[0] and fs[1]==env.fly_pos[1]) or
            (fs[2]==env.fly_pos[0] and fs[3]==env.fly_pos[1])
        )

        cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))

        lhs[:d] -= Phi[ns_idx]

        A_ub.append(lhs)
        b_ub.append(cost)

    # 🔥 FIX: single anchor constraint
    start_state = np.array([0,0,0,grid-1])
    start_idx = get_idx(start_state)

    row = np.zeros(d+1)
    row[:d] = Phi[start_idx]

    A_eq = [row]
    b_eq = [1.0]

    return np.array(A_ub), np.array(b_ub), np.array(A_eq), np.array(b_eq)