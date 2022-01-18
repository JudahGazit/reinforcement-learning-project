import gym
import matplotlib
import matplotlib.animation as animation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio

from model import Model

def play_by_model(env, model, batch_frames, file_name='game.gif'):
    state = env.reset()
    state = np.hstack([state for _ in range(batch_frames)])
    frames = []
    actions = []
    rewards = []
    total_reward = 0
    for i in range(500):
        action = model.act(state.reshape(1, -1))[0]
        batch_states = np.array([env.step(model.action_space[action]) for _ in range(batch_frames)])
        state, reward, done, info = np.concatenate(batch_states[:, 0]), *batch_states[-1, 1:]
        reward = np.maximum(reward, -10)
        reward = reward if reward < 1 else (10 * reward) ** 2
        total_reward += reward

        rewards.append(reward)
        actions.append(action)
        frames.append(env.render(mode='rgb_array'))
        if done:
            break

    imageio.mimsave(file_name, frames, duration=0.05)

    plt.plot(np.arange(len(rewards)), rewards, '-o')
    plt.title(f'reward per action - {file_name}')
    plt.show()

    plt.plot(np.arange(len(rewards)), np.cumsum(rewards), '-o')
    plt.title(f'reward per action - cumsum - {file_name}')
    plt.show()

    counts = pd.value_counts(actions).reset_index()
    counts['index'] = counts['index'].apply(lambda x: model.action_space[x])
    counts.set_index('index').head(10).plot(kind='barh', ax=plt.figure(figsize=(15, 5)).gca())
    plt.title(f'action dist - {file_name}')
    plt.show()

    # ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=False)
    # return ani


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3").env
    batch_frames = 2
    model = Model("BipedalWalker-v3", batch_frames).train()
    for i in range(10):
        play_by_model(env, model, batch_frames, f'game_temp{i+1}.gif')
    # animation.save('game.gif')