import gym
import keras.models
import tensorflow as tf
import matplotlib
import matplotlib.animation as animation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio

from model import Model


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def play_by_model(env, model, batch_frames, file_name='game.gif'):
    state = env.reset()
    state = np.hstack([state for _ in range(batch_frames)])
    frames = []
    actions = []
    rewards = []
    total_reward = 0
    for i in range(2000):
        action = model.act(state.reshape(1, -1), stochastic=False)[0]
        batch_states = np.array([env.step(action) for _ in range(batch_frames)])
        state, reward, done, info = np.concatenate(batch_states[:, 0]), np.array(batch_states[:, 1]), *batch_states[-1, 2:]
        reward = np.maximum(reward, -10).sum()
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
    #
    counts = pd.value_counts((10 * np.array(actions)).round().tolist()).reset_index()
    # counts['index'] = counts['index'].apply(lambda x: model.action_space[x])
    counts.set_index('index').head(10).plot(kind='barh', ax=plt.figure(figsize=(15, 5)).gca())
    plt.title(f'action dist - {file_name}')
    plt.show()

    # ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=False)
    # return ani

import json
def load_model(file_name='50k'):
    model = Model(env_name, batch_frames)
    model.model = keras.models.load_model(f'weights{file_name}.h5')
    model.target_model = keras.models.load_model(f'weights{file_name}.h5')
    j = json.load(open(f'params{file_name}.json'))
    for k, v in j.items():
        if '[' in v:
            v = np.fromstring(v[1:-1], sep=' ')
        elif float(v).is_integer():
            v = int(v)
        else:
            v = float(v)
        setattr(model.replay_memory, k, v)
    return model

if __name__ == '__main__':
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name).env
    batch_frames = 2
    with tf.device('cpu'):
        model, _ = Model(env_name, batch_frames).train(3000, 1000)
    for i in range(10):
        play_by_model(env, model, batch_frames, f'game_temp{i+1}.gif')
    # animation.save('game.gif')


