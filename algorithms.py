# Approximate Linear programming for solving SSP

from environment import SpiderFlyEnv
import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm


def approximate_linear_programming(env, max_iterations=100):

    agents = env.num_agents
    grid = env.grid_size

    feature_dimension = agents + 1


    # Initialize policy and weights
    policy = [np.random.randint(0, env.actions, size=(grid, grid, grid, grid)) for _ in range(agents)]

    weights = np.zeros(feature_dimension)


    # Pre-calculate Feature Matrix Phi

    num_states = grid ** (2*agents)

    Phi = np.zeros((num_states, feature_dimension))


    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                for l in range(grid):

                    idx = i*grid**3 + j*grid**2 + k*grid + l

                    Phi[idx,0] = abs(i - env.fly_pos[0]) + abs(j - env.fly_pos[1])
                    Phi[idx,1] = abs(k - env.fly_pos[0]) + abs(l - env.fly_pos[1])
                    Phi[idx,2] = 1.0


    def get_idx(s):

        # FIX: cleaner indexing
        return s[0]*grid**3 + s[1]*grid**2 + s[2]*grid + s[3]


    # FIX: proper ALP objective (state relevance weights)
    c = np.ones(num_states) / num_states
    objective = -(Phi.T @ c)


    for iteration in tqdm(range(max_iterations), desc="ALP Iterations"):

        A_ub, b_ub, A_eq, b_eq = [], [], [], []


        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    for l in range(grid):

                        state_idx = i*grid**3 + j*grid**2 + k*grid + l


                        # 1. IDENTIFY CAPTURE (terminal states)

                        if (i == env.fly_pos[0] and j == env.fly_pos[1]) or \
                           (k == env.fly_pos[0] and l == env.fly_pos[1]):

                            A_eq.append(Phi[state_idx])
                            b_eq.append(0)

                            continue 


                        curr_s = np.array([i, j, k, l])


                        # we are using simulate_step to get next state and cost for both spiders together to avoid mutating env.state and ensure correct collision handling
                        mid_s, c1 = env.simulate_step(curr_s, policy[0][i,j,k,l], 0)

                        final_s, c2 = env.simulate_step(mid_s, policy[1][i,j,k,l], 1)


                        next_idx = get_idx(final_s)


                        is_cap = (final_s[0] == env.fly_pos[0] and final_s[1] == env.fly_pos[1]) or \
                                 (final_s[2] == env.fly_pos[0] and final_s[3] == env.fly_pos[1])


                        # SSP uses COST not reward
                        cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))


                        # Bellman inequality: J(s) ≤ g + J(s')
                        A_ub.append(Phi[state_idx] - Phi[next_idx])

                        b_ub.append(cost)



        res = linprog(
            objective,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            method="highs"
        )


        if not res.success: #check if optimization succeeded

            print("LP Solver failed:", res.message)
            break


        weights = res.x.copy()


        # DECENTRALIZED POLICY IMPROVEMENT (1-step lookahead)


        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    for l in range(grid):

                        curr_s = np.array([i, j, k, l])


                        # Spider 1 improvement

                        best_q1 = float("inf")   #  cost minimization
                        best_a1 = policy[0][i,j,k,l]


                        for a1 in range(env.actions):
                            # mahor fix: simulate both steps together to get final state and cost and avoid mutating env.state
                            ms, c1 = env.simulate_step(curr_s, a1, 0)
                            
                            fs, c2 = env.simulate_step(ms, policy[1][i,j,k,l], 1)


                            is_cap = (fs[0] == env.fly_pos[0] and fs[1] == env.fly_pos[1]) or \
                                     (fs[2] == env.fly_pos[0] and fs[3] == env.fly_pos[1])


                            cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))


                            q = cost + np.dot(Phi[get_idx(fs)], weights)


                            if q < best_q1:

                                best_q1 = q
                                best_a1 = a1


                        policy[0][i,j,k,l] = best_a1


                        # Spider 2 improvement

                        best_q2 = float("inf")  
                        best_a2 = policy[1][i,j,k,l]

                        # simulate both steps together to get final state and cost and avoid mutating env.state
                        for a2 in range(env.actions):

                            ms, c1 = env.simulate_step(curr_s, policy[0][i,j,k,l], 0)

                            fs, c2 = env.simulate_step(ms, a2, 1)


                            is_cap = (fs[0] == env.fly_pos[0] and fs[1] == env.fly_pos[1]) or \
                                     (fs[2] == env.fly_pos[0] and fs[3] == env.fly_pos[1])


                            cost = 0 if is_cap else (1 + (2 if (c1 or c2) else 0))


                            q = cost + np.dot(Phi[get_idx(fs)], weights)

                            # minimize cost hence < best_q2 
                            if q < best_q2:

                                best_q2 = q
                                best_a2 = a2


                        policy[1][i,j,k,l] = best_a2


    return policy, weights
