import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from model import Model


def play_by_model(env, model, batch_frames):
    state = env.reset()
    state = np.concatenate([state, np.zeros((batch_frames - 1) * len(state))])
    frames = []
    total_reward = 0
    fig = plt.figure("Animation")
    ax = fig.add_subplot(111)
    for i in range(500):
        action = model.act(state.reshape(1, -1), stochastic=False)[0]
        batch_states = np.array([env.step(model.action_space[action]) for _ in range(batch_frames)])
        state, reward, done, info = np.concatenate(batch_states[:, 0]), *batch_states[-1, 1:]
        total_reward += reward

        frames.append([ax.imshow(env.render(mode='rgb_array')),
                       ax.annotate(f'{state[0]:.4f}, {reward}', (0.5, 0.9), xycoords='figure fraction')])

        if done:
            break
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=False)
    return ani


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3").env
    batch_frames = 1
    model = Model(env, batch_frames).train()
    animation = play_by_model(env, model, batch_frames)
    animation.save('game.gif')