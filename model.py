import itertools

import gym
import keras.optimizer_v2.adam
import numpy as np
import random
import warnings
from collections import deque
import copy

import numpy as np
import scipy.special
from keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate
from keras.models import Sequential
from keras.regularizers import l2

import matplotlib.pyplot as plt
import shelve

warnings.filterwarnings("ignore")

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.mean = 0
        self.std = 1

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size

class Model:
    def __init__(self, env, batch_frames, action_step_size=5):
        self.env_name = env
        self.env = gym.make(self.env_name).env

        self.parallel_envs = 1
        self.action_step_size = action_step_size
        self.copy_to_target_at = 500
        self.minimum_states = 5000
        self.batch_frames = batch_frames
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

        self.action_space = self.__make_action_space()

        # self.replay_memory_pickle = shelve.open('replay.pickle', writeback=True)
        # if 'mem' not in self.replay_memory_pickle:
        #     self.replay_memory_pickle['mem'] = ReplayMemory(500_000)
        # self.replay_memory = self.replay_memory_pickle['mem']
        self.replay_memory = ReplayMemory(500_000)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.copy_to_target()


    def __make_action_space(self):
        number_of_axes = self.env.action_space.shape[0]
        # actions = (np.linspace(-1, 1, self.action_step_size) * np.eye(number_of_axes)[:, :, None]).reshape(number_of_axes, -1, ).transpose()
        actions = itertools.product(*[(-1, 0.5, 0, 0.5, 1) for _ in range(number_of_axes)])
        actions = np.array(list(actions))
        # actions = actions[((actions == 0).sum(1) >= 2) & ((actions == 0).sum(1) < number_of_axes)]
        actions = actions[((actions != 0).sum(1) <= 1) & ((actions != 0).sum(1) > 0)]
        # actions = np.concatenate([actions, np.zeros((1, number_of_axes))])
        print('Number of actions', len(actions))
        return actions

    def create_model(self):
        model = Sequential()
        number_of_features = self.env.observation_space.shape[0] * self.batch_frames
        print('Number of features', number_of_features)
        model.add(Input(number_of_features))
        for _ in range(3):
            model.add(Dense(256))
            # model.add(Dropout(0.2))
        model.add(Dense(len(self.action_space)))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizer_v2.adam.Adam(learning_rate=self.learning_rate))
        return model

    def copy_to_target(self):
        self.target_model.set_weights(copy.deepcopy(self.model.get_weights()))

    def remember(self, state, action, reward, new_state, done):
        self.replay_memory.append([state, action, reward, new_state, done])

    def act(self, states, stochastic=True):
        # model_actions = self.model.predict(states)
        # try:
        #     action_prob = scipy.special.softmax(model_actions, 1)
        #     actions = np.array([np.random.choice(action_prob.shape[1], 1, p=p) for p in action_prob]).flatten()
        #     return actions
        # except:
        #     return model_actions.argmax(1)
        greedy_actions = np.argmax(self.model.predict((states - self.replay_memory.mean) / self.replay_memory.std, batch_size=len(states)), 1)
        actions = np.array([
                            np.random.choice(len(self.action_space)) if (np.random.random() < self.epsilon) and stochastic
                            else greedy_action
                            for greedy_action in greedy_actions
        ])
        return actions


    def learn_over_replay(self):
        if len(self.replay_memory) >= self.minimum_states:
            samples = np.array(self.replay_memory.sample(self.batch_size))
            rewards = samples[:, 2]
            is_done = samples[:, 4]
            state = (np.vstack(samples[:, 0]) - self.replay_memory.mean) / self.replay_memory.std
            action = samples[:, 1].astype(int)
            next_state = (np.vstack(samples[:, 3]) - self.replay_memory.mean) / self.replay_memory.std

            targets = self.model.predict(state, batch_size=len(state))
            next_action = self.model.predict(next_state, batch_size=len(state)).argmax(1)
            Q_future = self.target_model.predict(next_state, batch_size=len(state))[np.arange(len(targets)), next_action]
            targets[np.arange(len(targets)), action] = rewards + (1 - is_done) * Q_future * self.gamma

            loss = self.model.fit(state, targets, epochs=1, verbose=False, batch_size=self.batch_size, shuffle=False)
            return loss.history['loss'][0]
        else:
            self.replay_memory.mean = np.array(self.replay_memory.buffer[:self.replay_memory.size])[:, 0].mean(0)
            self.replay_memory.std = np.array(self.replay_memory.buffer[:self.replay_memory.size])[:, 0].std(0)
        return 0

    def train(self, episodes=1000, episode_length=700):
        envs = [gym.make(self.env_name) for _ in range(self.parallel_envs)]
        losses = []
        total_rewards = []
        for trial in range(episodes):
            cur_states = np.array([env.reset() for env in envs])
            cur_states = np.hstack([cur_states for _ in range(self.batch_frames)])
            total_reward = 0
            losses_of_trial = []
            for step in range(episode_length):
            # while True:
                if len(cur_states) > 0:
                    actions = self.act(cur_states)
                    new_states = np.array([[env.step(self.action_space[action]) for _ in range(self.batch_frames)]
                                           for env, action in zip(envs, actions)])
                    rewards = np.maximum(new_states[:, :, 1].mean(1), -10)
                    # rewards = rewards * (rewards < 0) + (1 * rewards * (rewards > 0)) - 0.01
                    total_reward += rewards.mean()
                    for state, action, reward, new_state in zip(cur_states, actions, rewards, new_states):
                        self.remember(state, action, reward, np.hstack(new_state[:, 0]), new_state[:, 2].any())
                    loss = self.learn_over_replay()
                    losses_of_trial.append(loss)

                    cur_states = np.array(new_states[:, :, 0].tolist()).reshape(cur_states.shape)
                    # cur_states = np.delete(cur_states, np.where(new_states[:, :, 2].any(1))[0], axis=0)
                    if new_states[:, :, 2].any():
                        break
            self.copy_to_target()
                # for i in np.where(new_states[:, :, 2].any(1))[0]:
                #     reset_game = envs[i].reset()
                #     cur_states[i] = np.hstack([reset_game for _ in range(self.batch_frames)])
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # if trial % (self.copy_to_target_at // self.parallel_envs) == 0:
            #     self.copy_to_target()
            losses.append(np.mean(losses_of_trial))
            total_rewards.append(total_reward)
            print('Episode', str.zfill(str(trial), 4), '\t|\t',
                  'Reward', f'{total_reward:.4f}', '\t|\t',
                  'Queue', len(self.replay_memory), '\t|\t',
                  'Epsilon', f'{self.epsilon:.4f}', '\t|\t',
                  'Loss', f'{np.mean(losses_of_trial):.4f}', '\t|\t',
                  # 'TargetUpdates', trial // (self.copy_to_target_at // self.parallel_envs), '\t|\t',
                  sep='\t')
        plt.plot(range(episodes), losses, '-')
        plt.setp(plt.gca(), title='Losses per episode', xlabel='episode', ylabel='mean loss')
        plt.show()

        plt.plot(range(episodes), total_rewards, '-')
        plt.setp(plt.gca(), title='Rewards per episode', xlabel='episode', ylabel='total reward')
        plt.show()
        return self