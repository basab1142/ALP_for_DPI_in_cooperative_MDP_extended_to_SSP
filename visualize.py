import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import SpiderFlyEnv
from algorithms import approximate_linear_programming
def rollout_episode(env, policy, max_steps=50):

    state = env.reset()
    trajectory = [state.copy()]

    for _ in range(max_steps):

        i, j, k, l = state

        a1 = policy[0][i,j,k,l]
        a2 = policy[1][i,j,k,l]

        # spider 1 moves
        mid_state, _ = env.simulate_step(state, a1, 0)

        # spider 2 moves
        next_state, _ = env.simulate_step(mid_state, a2, 1)

        trajectory.append(next_state.copy())

        state = next_state

        # stop if fly captured
        if (state[0],state[1]) == env.fly_pos or (state[2],state[3]) == env.fly_pos:
            break

    return trajectory






def animate_trajectory(env, trajectory, save_path="trajectory_animation.gif"):

    grid = env.grid_size

    fig, ax = plt.subplots()

    ax.set_xlim(-0.5, grid-0.5)
    ax.set_ylim(-0.5, grid-0.5)

    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))

    ax.grid(True)

    fly_x, fly_y = env.fly_pos

    fly_plot, = ax.plot(fly_x, fly_y, "ro", markersize=12, label="Fly")
    spider1_plot, = ax.plot([], [], "bo", markersize=12, label="Spider 1")
    spider2_plot, = ax.plot([], [], "go", markersize=12, label="Spider 2")

    ax.legend()


    def update(frame):

        state = trajectory[frame]

        s1x, s1y, s2x, s2y = state

        spider1_plot.set_data([s1x], [s1y])
        spider2_plot.set_data([s2x], [s2y])

        return spider1_plot, spider2_plot


    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=500,
        repeat=False,
        blit=False
    )

    # Save animation as GIF using Pillow
    from matplotlib.animation import PillowWriter
    ani.save(save_path, writer=PillowWriter(fps=2))

    print(f"Animation saved to {save_path}")

    return ani

env = SpiderFlyEnv()

# load policy and weights 
policy, weights = np.load("policy.npy", allow_pickle=True), np.load("weights.npy", allow_pickle=True)

trajectory = rollout_episode(env, policy)

ani = animate_trajectory(env, trajectory)
