# Approximate Linear programming for solving SSP
from environment import SpiderFlyEnv
import numpy as np
from scipy.optimize import linprog
# # INITIALIZATION
# Initialize policy mu for all spiders (e.g., greedy move toward fly) 
# Initialize weights r randomly or to zeros
# Define Feature Matrix Phi for all possible states (Manhattan distances) 

# # MAIN LOOP (Policy Iteration)
# repeat:
#     # --- PHASE 1: APPROXIMATE POLICY EVALUATION (Algorithm 4) ---
#     # Find weight vector 'r' for the current policy 'mu'
    
#     Solve Linear Program (LP):
#         Objective: Maximize sum(Phi * r)  # Push values as high as possible 
        
#         Constraints for every state 'x':
#             # Bellman Inequality (SSP version: alpha = 1)
#             Phi(x) * r <= Reward(x, mu(x)) + Phi(next_x) * r 
            
#             # The Goal Anchor (SSP Essential)
#             Phi(fly_pos) * r = 0  # Forces goal cost to zero
            
#     r_updated = results of LP solver 
    
#     # --- PHASE 2: DECENTRALIZED POLICY IMPROVEMENT (Algorithm 1) ---
#     # Agents take turns updating moves using the scorecard 'r_updated'
    
#     for agent i in [1, 2]:
#         for every state 'x':
#             # Find best action assuming other agent's move is fixed
#             # Evaluate: Reward(x, u_i, other_u) + Phi(next_x) * r_updated
#             new_action = argmin(look_ahead_cost) 
#             mu[agent i](x) = new_action
            
# until policy mu or weights r stop changing significantly

def approximate_linear_programming(env=SpiderFlyEnv, max_iterations=100, tolerance=1e-4):
    # initialize policy and weights
    
    agents = env.num_agents
    feature_dimesnion = env.num_agents  + 1  # distance for each agent and bias
    # For simplicity, we will use a random policy for both agents and zero weights
    policy = [np.random.randint(0, env.actions, size=(env.grid_size, env.grid_size, env.grid_size, env.grid_size))] * agents  
    weights = np.zeros(feature_dimesnion)  # weights for features
    #  feature matrix Phi (using Manhattan distance to fly as a feature) size: ((grid_sizex)^(2*agents)xfeature_dimension)
      # each agents distance and one for bias
    Phi = np.zeros((env.grid_size**(2*agents), feature_dimesnion))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            for k in range(env.grid_size):
                for l in range(env.grid_size):
                    state_index = i*env.grid_size**3 + j*env.grid_size**2 + k*env.grid_size + l
                    # features: distance of each agent to fly and bias
                    Phi[state_index, 0] = abs(i - env.fly_pos[0]) + abs(j - env.fly_pos[1])  # distance for agent 1
                    Phi[state_index, 1] = abs(k - env.fly_pos[0]) + abs(l - env.fly_pos[1])  # distance for agent 2
                    Phi[state_index, 2] = 1  # bias term
    # now solve the LP for policy evaluation and update policy iteratively
    for iteration in range(max_iterations):
        # define c = -Phi.T @ np.ones(Phi.shape[0]) to maximize sum of values
        c = -Phi.T @ np.ones(Phi.shape[0])
        # A_ub = phi[current_state] - gamma * phi[next_state]
        A_ub = []
        # b_ub = reward(current_state, policy) for all states
        b_ub = []
        # A_eq for goal anchor constraint
        A_eq = []
        # b_eq for goal anchor constraint
        b_eq = []
        #TODO: its wrong I need to solve it
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                for k in range(env.grid_size):
                    for l in range(env.grid_size):
                        current_state = np.array([i, j, k, l])
                        state_index = i*env.grid_size**3 + j*env.grid_size**2 + k*env.grid_size + l
                        # get action from policy for both agents
                        action1 = policy[0][i, j, k, l]
                        action2 = policy[1][i, j, k, l]
                        # simulate environment step to get next state and reward
                        env.state = current_state.copy()
                        next_state1, reward1, _ = env.step(action1, 0)
                        next_state2, reward2, _ = env.step(action2, 1)
                        # Bellman inequality constraints for both agents
                        A_ub.append(Phi[state_index] - Phi[next_state1[0]*env.grid_size**3 + next_state1[1]*env.grid_size**2 + next_state1[2]*env.grid_size + next_state1[3]])
                        b_ub.append(reward1)
                        A_ub.append(Phi[state_index] - Phi[next_state2[0]*env.grid_size**3 + next_state2[1]*env.grid_size**2 + next_state2[2]*env.grid_size + next_state2[3]])
                        b_ub.append(reward2)
        # Goal anchor constraint: phi(any of fly_pos == fly_pos) * r = 0
       
        fly_state_index = env.fly_pos[0]*env.grid_size**3 + env.fly_pos[1]*env.grid_size**2 + env.fly_pos[0]*env.grid_size + env.fly_pos[1]
        A_eq.append(Phi[fly_state_index])
    
