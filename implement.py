from environment import SpiderFlyEnv
from algorithms import approximate_linear_programming
import numpy as np
if __name__ == "__main__":
    env = SpiderFlyEnv()
    policy, weights = approximate_linear_programming(env)
    # save weights and policy to file for later use
    np.save("weights.npy", weights)
    np.save("policy.npy", policy)