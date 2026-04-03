import numpy as np

# spider-fly environment
# spider will start at (0,0) and (0,3), fly will start at (3,3), spiders will move simultaneously
# actions: 0:up, 1:down, 2:left, 3:right
# fly remain stationary, so we can ignore it in the state representation
# for every step, there is a step reward of -1 and when fly is captured, reward is 0.
# So the cost is the distance to the fly, and the goal is to minimize the distance to the fly.
# we will formulate it as SSP, so no discounting, and the episode ends when the fly is captured.
# But we can also add a discount factor for infinite horizon formulation.
# 2 spiders cannot be in the same cell, so we can add a constraint that if they try to move
# into the same cell, they will collide and stay in their original position.


class SpiderFlyEnv:

    def __init__(self, grid_size=4, num_spiders=2, fly_pos=(3,3), discount=1.0):

        self.grid_size = grid_size
        self.num_spiders = num_spiders
        self.fly_pos = fly_pos
        self.num_agents = num_spiders  # spiders only
        self.discount = discount
        self.actions = 4  # 0:up, 1:down, 2:left, 3:right

        self.reset()


    def reset(self):

        # Spider1 at (0,0), Spider2 at (0,3)
        self.state = np.array([0,0, 0,3])  # only spiders' positions
        return self.state


    def step(self, action, agent_number):

        current_state = self.state.copy()

        x, y = current_state[agent_number*2 : agent_number*2 + 2]

        other_idx = 1 - agent_number
        other_pos = current_state[other_idx*2 : other_idx*2 + 2]


        # Calculate new position (without blocking)

        if action == 0: # up
            y = min(y + 1, self.grid_size - 1)

        elif action == 1: # down
            y = max(y - 1, 0)

        elif action == 2: # left
            x = max(x - 1, 0)

        elif action == 3: # right
            x = min(x + 1, self.grid_size - 1)


        collided = np.array_equal([x, y], other_pos)

        # collision blocks movement (otherwise spiders overlap)
        if collided:
            x, y = current_state[agent_number*2 : agent_number*2 + 2]


        current_state[agent_number*2 : agent_number*2 + 2] = [x, y]

        self.state = current_state


        reward = self.get_reward(collided)

        done = (reward == 0)

        return self.state, reward, done


    def get_reward(self, collided=False):

        # Check if any spider is on the fly
        for i in range(self.num_spiders):

            if np.array_equal(self.state[i*2:(i+1)*2], self.fly_pos):
                return 0  


        reward = -1

        if collided:
            reward -= 10

        return reward


    def simulate_step(self, state, action, agent_number):

        # simulate_step method that predicts the next state
        # without actually changing self.state

        current_state = state.copy()

        x, y = current_state[agent_number*2 : agent_number*2 + 2]

        other_idx = 1 - agent_number
        other_pos = current_state[other_idx*2 : other_idx*2 + 2]


        if action == 0:
            y = min(y + 1, self.grid_size - 1)

        elif action == 1:
            y = max(y - 1, 0)

        elif action == 2:
            x = max(x - 1, 0)

        elif action == 3:
            x = min(x + 1, self.grid_size - 1)


        collided = np.array_equal([x, y], other_pos)


        # collision blocks movement
        if collided:
            x, y = current_state[agent_number*2 : agent_number*2 + 2]


        current_state[agent_number*2 : agent_number*2 + 2] = [x, y]


        return current_state, collided

    def simulate_joint_step(self, state, a1, a2):

        s = state.copy()

        # compute intended moves
        def move(pos, action):
            x, y = pos
            if action == 0:
                y = min(y + 1, self.grid_size - 1)
            elif action == 1:
                y = max(y - 1, 0)
            elif action == 2:
                x = max(x - 1, 0)
            elif action == 3:
                x = min(x + 1, self.grid_size - 1)
            return np.array([x, y])

        p1 = move(s[0:2], a1)
        p2 = move(s[2:4], a2)

        # collision: both try same cell
        if np.array_equal(p1, p2):
            return s.copy(), True, True

        # otherwise both move
        new_state = np.array([p1[0], p1[1], p2[0], p2[1]])

        c1 = np.array_equal(p1, s[2:4])
        c2 = np.array_equal(p2, s[0:2])

        return new_state, c1, c2