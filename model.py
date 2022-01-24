import copy
import itertools
import random
import warnings

import gym
import keras.losses
import keras.models
import keras.optimizers
import keras.optimizer_v2.adam
import keras.optimizer_v2.rmsprop
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, GaussianNoise, ReLU
from mlflow import log_metric, log_param

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
        return random.sample(self.buffer[:self.size], batch_size)

    def update_mead_std(self):
        states = np.array(self.buffer[:self.size])[:, 0]
        self.mean = np.mean(states, 0)
        self.std = np.std(states, 0)

    def __len__(self):
        return self.size

class Model:
    def __init__(self, env, batch_frames, action_step_size=3):
        self.env_name = env
        self.env = gym.make(self.env_name).env

        self.parallel_envs = 1
        self.action_step_size = action_step_size
        self.copy_to_target_at = 10
        self.learn_every = 2
        self.minimum_states = 5000
        self.batch_frames = batch_frames
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.learning_rate_decay = 1e-2

        for k, v in self.__dict__.items():
            if k != 'env':
                log_param(k ,v)

        self.action_space = self.__make_action_space()

        self.replay_memory = ReplayMemory(500_000)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.copy_to_target()


    def __make_action_space(self):
        number_of_axes = self.env.action_space.shape[0]
        actions = itertools.product(*[np.linspace(-1, 1, self.action_step_size) for _ in range(number_of_axes)])
        actions = np.array(list(actions))
        actions = actions[(np.sum(actions != 0, 1) <= 4) & (np.sum(actions != 0, 1) > 0)]
        print('Number of actions', len(actions))
        log_param('num_actions', len(actions))
        log_param('max_parallel_actions', np.max(np.sum(actions != 0, 1)))
        return actions

    def create_model(self):
        number_of_features = self.env.observation_space.shape[0] * self.batch_frames
        print('Number of features', number_of_features)
        X_input = Input(number_of_features)
        X = X_input
        for _ in range(3):
            X = Dense(256, activation='relu')(X)
        # X = GaussianNoise(0.001)(X)
        advantage = Dense(len(self.action_space))(X)
        value = Dense(1)(X)
        X = value + (advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True))
        model = keras.Model(inputs=X_input, outputs=X)
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizer_v2.rmsprop.RMSprop(learning_rate=self.learning_rate))
        return model

    def copy_to_target(self):
        self.target_model.set_weights(copy.deepcopy(self.model.get_weights()))

    def remember(self, state, action, reward, new_state, done):
        self.replay_memory.append([state, action, reward, new_state, done])

    def act(self, states, stochastic=True):
        greedy_actions = np.argmax(self.model.predict((states - self.replay_memory.mean) / self.replay_memory.std, batch_size=len(states)), 1)
        actions = np.array([
                            random.randrange(len(self.action_space)) if (np.random.random() < self.epsilon) and stochastic
                            else greedy_action
                            for greedy_action in greedy_actions
        ])
        return actions


    def create_targets(self, state, action, reward, new_state, done):
        targets = self.model.predict(state, batch_size=len(state))
        next_action = self.model.predict(new_state, batch_size=len(state)).argmax(1)
        Q_future = self.target_model.predict(new_state, batch_size=len(state))[np.arange(len(targets)), next_action]
        targets[np.arange(len(targets)), action] = reward + (1 - done) * Q_future * self.gamma
        return targets

    def learn_over_replay(self):
        if len(self.replay_memory) >= self.minimum_states:
            samples = np.array(self.replay_memory.sample(self.batch_size))
            rewards = samples[:, 2]
            is_done = samples[:, 4]
            state = (np.vstack(samples[:, 0]) - self.replay_memory.mean) / self.replay_memory.std
            action = samples[:, 1].astype(int)
            next_state = (np.vstack(samples[:, 3]) - self.replay_memory.mean) / self.replay_memory.std

            targets = self.create_targets(state, action, rewards, next_state, is_done)

            loss = self.model.fit(state, targets, epochs=1, verbose=False, batch_size=self.batch_size, shuffle=False)
            return loss.history['loss'][0]
        else:
            self.replay_memory.update_mead_std()
        return 0

    def train(self, episodes=1000, episode_length=2000):
        envs = [gym.make(self.env_name).env for _ in range(self.parallel_envs)]
        losses = []
        total_rewards = []
        for trial in range(episodes):
            cur_states = np.array([env.reset() for env in envs])
            cur_states = np.hstack([cur_states for _ in range(self.batch_frames)])
            total_reward = 0
            losses_of_trial = []
            for step in range(episode_length):
                if len(cur_states) > 0:
                    actions = self.act(cur_states)
                    new_states = np.array([[env.step(self.action_space[action]) for _ in range(self.batch_frames)]
                                           for env, action in zip(envs, actions)])
                    rewards = np.maximum(new_states[:, :, 1], -10).sum(1)
                    total_reward += rewards.mean()
                    for state, action, reward, new_state in zip(cur_states, actions, rewards, new_states):
                        self.remember(state, action, reward, np.hstack(new_state[:, 0]), new_state[:, 2].any())
                    if step % self.learn_every == 0:
                        loss = self.learn_over_replay()
                        losses_of_trial.append(loss)

                    cur_states = np.array(new_states[:, :, 0].tolist()).reshape(cur_states.shape)
                    if new_states[:, :, 2].any():
                        break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if trial % (self.copy_to_target_at // self.parallel_envs) == 0:
                self.copy_to_target()
                self.replay_memory.update_mead_std()

            losses.append(np.mean(losses_of_trial))
            total_rewards.append(total_reward)
            log_metric('mean_episode_loss', np.mean(losses_of_trial), step=trial)
            log_metric('epsilon', self.epsilon, step=trial)
            log_metric('queue_size', len(self.replay_memory), step=trial)
            log_metric('reward', total_reward, step=trial)
            print('Episode', str.zfill(str(trial), 4), '\t|\t',
                  'Reward', f'{total_reward:.4f}', '\t|\t',
                  'Queue', len(self.replay_memory), '\t|\t',
                  'Epsilon', f'{self.epsilon:.4f}', '\t|\t',
                  'Loss', f'{np.mean(losses_of_trial):.4f}', '\t|\t',
                  # 'TargetUpdates', trial // (self.copy_to_target_at // self.parallel_envs), '\t|\t',
                  sep='\t')
        self.model.save('weights')
        return self