from environment import SpiderFlyEnv
from algorithms import approximate_linear_programming
import numpy as np
import matplotlib.pyplot as plt

def plot_policy(policy, grid, agent_id, fixed_other=(0, 0)):
    arrows = {
        0: (0, 1),  # up
        1: (0, -1),  # down
        2: (-1, 0),  # left
        3: (1, 0)  # right
    }

    plt.figure()

    X, Y = np.meshgrid(range(grid), range(grid))

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(grid):
        for j in range(grid):

            if agent_id == 0:
                a = policy[0][i, j, fixed_other[0], fixed_other[1]]
            else:
                a = policy[1][fixed_other[0], fixed_other[1], i, j]

            dx, dy = arrows[a]
            U[j, i] = dx
            V[j, i] = dy

    plt.quiver(X, Y, U, V)
    plt.title(f"Agent {agent_id} Policy")
    # plt.gca().invert_yaxis()
    plt.savefig("policy_agent" + str(agent_id) + ".png")

if __name__ == "__main__":
    env = SpiderFlyEnv()
    policy, weights = approximate_linear_programming(env)
    # save weights and policy to file for later use
    np.save("weights.npy", weights)
    np.save("policy.npy", policy)

    plot_policy(policy, env.grid_size, agent_id=0)
    plot_policy(policy, env.grid_size, agent_id=1)