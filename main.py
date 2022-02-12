import logging
import sys

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dddqn.agent import Agent

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stdout, level=logging.INFO)

def play_by_model(model, file_name='game.gif'):
    rewards, frames = model.play(2000, render=True)

    imageio.mimsave(file_name, frames, fps=30)

    plt.plot(np.arange(len(rewards)), rewards, '-o')
    plt.title(f'reward per action - {file_name}')
    plt.show()

    plt.plot(np.arange(len(rewards)), np.cumsum(rewards), '-o')
    plt.title(f'reward per action - cumsum - {file_name}')
    plt.show()

def grade_model(model, trials=100):
    trial_scores = []
    for _ in tqdm(range(trials)):
        trial_scores.append(np.sum(model.play(2000, render=False)))

    plt.hist(trial_scores, 10)
    plt.setp(plt.gca(), title=f'{model.env_name} - Rewards of {trials} trials')
    plt.show()
    return np.mean(trial_scores)

if __name__ == '__main__':
    env_name = "BipedalWalkerHardcore-v3"
    env = gym.make(env_name).env
    batch_frames = 2
    # with tf.device('cpu'):
    #     model = Agent(env_name, batch_frames).train(3000, 1000)
    #     model.save('saved_models/hardcore_weights_clip_state')
    model = Agent(env_name, batch_frames).load('saved_models/hardcore_weights')
    print('Model score', grade_model(model))
    for i in range(10):
        play_by_model(model, f'gifs/game_temp{i+1}.gif')
