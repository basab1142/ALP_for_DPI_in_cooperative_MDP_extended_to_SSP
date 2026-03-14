import numpy as np
import cvxpy as cp
from environment import SpiderFlyEnv

env = SpiderFlyEnv()
states = env.get_all_states()
n_states = len(states)
d_features = len(env.get_features(states[0]))

# Φ matrix
Phi = np.array([env.get_features(s) for s in states])

def get_policy_action(policy, state, agent):
    # policy is list of dicts, policy[i][state] = action for agent i
    return policy[agent].get(tuple(state), 0)  # default 0

def CACFN(policy, alpha):
    # Algorithm 4
    # Solve LP: max 1^T (Phi r) s.t. Phi r <= T_mu Phi r
    r = cp.Variable(d_features)
    constraints = []
    for idx, x in enumerate(states):
        x_tuple = tuple(x)
        # Compute T_mu Phi r (x)
        t_mu = 0
        # Since deterministic, but policy may be stochastic, but assume deterministic
        actions = [get_policy_action(policy, x_tuple, i) for i in range(env.num_agents)]
        next_state, cost, _ = env.step(actions)
        next_idx = np.where((states == next_state).all(axis=1))[0][0]
        t_mu = cost + alpha * (Phi[next_idx] @ r)
        constraints.append(Phi[idx] @ r <= t_mu)
    objective = cp.Maximize(cp.sum(Phi @ r))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        print("LP not optimal")
        return np.zeros(d_features)
    return Phi @ r.value

def DPIm(i, updated_policies, base_policies, J, alpha):
    # Algorithm 1
    # For agent i, update policy
    new_policy = {}
    for x in states:
        x_tuple = tuple(x)
        best_a = None
        best_val = float('inf')
        for a_i in range(env.actions):
            actions = []
            for j in range(env.num_agents):
                if j < i:
                    actions.append(get_policy_action(updated_policies, x_tuple, j))
                elif j == i:
                    actions.append(a_i)
                else:
                    actions.append(get_policy_action(base_policies, x_tuple, j))
            env.state = x.copy()
            next_state, cost, _ = env.step(actions)
            next_idx = np.where((states == next_state).all(axis=1))[0][0]
            val = cost + alpha * J[next_idx]
            if val < best_val:
                best_val = val
                best_a = a_i
        new_policy[x_tuple] = best_a
    return new_policy

def DPI_ALP(T, alpha):
    # Algorithm 5
    # Initial policy: random
    policy = [{} for _ in range(env.num_agents)]
    for i in range(env.num_agents):
        for x in states:
            policy[i][tuple(x)] = np.random.randint(env.actions)
    
    J = CACFN(policy, alpha)
    for t in range(T):
        new_policies = []
        for i in range(env.num_agents):
            updated_policies = policy[:i] + [{}] + policy[i+1:]  # empty for i
            base_policies = policy[i+1:] + policy[:i]  # wait, adjust
            # Wait, in the pseudocode, ˜µ1:i−1, µi+1:m
            # So updated_policies = policy[:i], base_policies = policy[i:]
            # But policy[i] is the current for i
            # Wait, ˜µ1:i−1 are the updated for 1 to i-1, µi+1:m base for i+1 to m
            # For i, the others are: updated 1 to i-1, base i+1 to m
            # So updated_policies = policy[:i], base_policies = policy[i+1:]
            new_pi = DPIm(i, policy[:i], policy[i+1:], J, alpha)
            new_policies.append(new_pi)
        policy = new_policies
        J = CACFN(policy, alpha)
    return policy, J

# Run
policy, J = DPI_ALP(T=10, alpha=0.9)
print("Final policy:", policy)
print("Final J:", J)