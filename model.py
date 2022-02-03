import copy
import itertools
import json
import random
import warnings

import gym
import keras.losses
import keras.models
import keras.optimizers
import keras.optimizer_v2.adam
import keras.optimizer_v2.rmsprop
import numpy as np
import scipy.special
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Dense, Input, GaussianNoise, ReLU, BatchNormalization, Dropout
# from mlflow import log_metric, log_param
EPSILON = 1e-12

warnings.filterwarnings("ignore")

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.mean = 0
        self.mean_reward = 0
        self.std = 1
        self.std_reward = 1

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer[:self.size], batch_size)

    def update_mead_std(self):
        states = np.array(self.buffer[:self.size])[:, 0]
        rewards = np.array(self.buffer[:self.size])[:, 2]
        self.mean = np.mean(states, 0)
        self.std = np.std(states, 0)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)

    def __len__(self):
        return self.size

class Model:
    def __init__(self, env, batch_frames, action_step_size=3, tau=0.68, learning_rate=9e-05):
        self.env_name = env
        self.env = gym.make(self.env_name).env

        self.parallel_envs = 1
        self.action_step_size = action_step_size
        self.copy_to_target_at = 1 #copy_to_target_at
        self.learn_every = 8
        self.minimum_states = 5000
        self.batch_frames = batch_frames
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        # self.learning_rate = 0.00005478 #learning_rate ## Normal
        self.learning_rate = learning_rate
        self.tau = tau
        # self.tau = 0.8307
        # self.learning_rate = 0.0004386 # learning_rate ## hardcore
        # self.tau = 0.0662
        # self.learning_rate = 0.000164 # learning_rate ## hardcore

        # for k, v in self.__dict__.items():
        #     if k != 'env':
        #         log_param(k ,v)

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
        # log_param('num_actions', len(actions))
        # log_param('max_parallel_actions', np.max(np.sum(actions != 0, 1)))
        return actions

    def create_model(self):
        number_of_features = self.env.observation_space.shape[0] * self.batch_frames
        action_size = self.env.action_space.shape[0]
        print('Number of features', number_of_features)
        X_input = Input(number_of_features)
        action = Input(action_size)

        X = X_input
        X = Dense(512, activation='relu')(X)
        X = Dense(512, activation='relu')(X)
        X = Dense(256, activation='relu')(X)

        mu = Dense(action_size, activation='tanh', name='mu')(X)
        entries = Dense(action_size * (action_size + 1) / 2, activation='tanh')(X)
        V = Dense(1, name='V')(X)
        L = tfp.math.fill_triangular(entries)
        L = tf.linalg.set_diag(L, tf.exp(tf.linalg.diag_part(L)))
        P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))
        A = - 0.5 * tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(action, -1) - tf.expand_dims(mu, -1), (0, 2, 1)), P), tf.expand_dims(action, -1) - tf.expand_dims(mu, -1))
        Q = keras.layers.Add(name='Q')([tf.squeeze(A, 1), V])

        # dummy_loss = lambda y_true, y_pred: 0.0
        model = keras.Model(inputs=[X_input, action], outputs=[Q, V, mu])
        model.compile(loss='huber_loss',
                      optimizer=keras.optimizer_v2.rmsprop.RMSprop(learning_rate=self.learning_rate, clipnorm=1))
        return model

    def copy_to_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, new_state, done):
        payload = [state, action, reward, new_state, done]
        self.replay_memory.append(payload)
        return payload

    def act(self, states, stochastic=True):
        _, _, action = self.model.predict([(states - self.replay_memory.mean) / (self.replay_memory.std + EPSILON),
                                      np.zeros((len(states), 4))],
                                     batch_size=len(states))
        # action = np.clip(action + stochastic * self.epsilon * np.random.randn(action.shape[1]), -1, 1)
        if stochastic and random.random() < self.epsilon:
            action = np.random.random(action.shape) * 2 - 1
        return action



    def create_targets(self, state, action, reward, new_state, done):
        Q, V, _ = self.model.predict([state, action], batch_size=len(state))
        _, new_V, _ = self.target_model.predict([new_state, np.zeros((len(new_state), 4))], batch_size=len(state))
        discounted_rewards = reward + (1 - done) * new_V.flatten() * self.gamma
        targets = np.expand_dims(discounted_rewards, -1).astype(float), V, action
        return targets

    def learn_over_replay(self, stored):
        if len(self.replay_memory) >= self.minimum_states:
            samples = np.array(self.replay_memory.sample(self.batch_size))
            # samples = np.concatenate([self.replay_memory.sample(self.batch_size - 1), [stored]])
            state, action, rewards, next_state, is_done = [samples[:, i] for i in range(samples.shape[1])]
            state = (np.vstack(state) - self.replay_memory.mean) / (self.replay_memory.std + EPSILON)
            next_state = (np.vstack(next_state) - self.replay_memory.mean) / (self.replay_memory.std + EPSILON)
            action = np.vstack(action)
            rewards = (rewards / (self.replay_memory.std_reward + EPSILON)).astype(np.float)
            targets = self.create_targets(state, action, rewards, next_state, is_done)
            loss = self.model.fit([state, action], targets, epochs=1, verbose=False, batch_size=self.batch_size, shuffle=False)
            return loss.history['loss'][0]
        else:
            self.replay_memory.update_mead_std()
        return 0

    def train(self, episodes=3000, episode_length=2000):
        envs = [gym.make(self.env_name).env for _ in range(self.parallel_envs)]
        losses = []
        total_rewards = []
        total_steps = 0
        for trial in range(episodes):
            if len(total_rewards) > 5 and np.mean(total_rewards[-5:]) > 300:
                break
            states = np.array([env.reset() for env in envs])
            states = np.hstack([states for _ in range(self.batch_frames)])
            total_reward = 0
            losses_of_trial = []
            for step in range(episode_length):
                total_steps += 1
                actions = self.act(states)
                new_states = np.array([[env.step(action) for _ in range(self.batch_frames)]
                                       for env, action in zip(envs, actions)])
                rewards = np.maximum(np.maximum(new_states[:, :, 1], -10).sum(1), -10)
                total_reward += rewards.mean()
                for state, action, reward, new_state in zip(states, actions, rewards, new_states):
                    stored = self.remember(state, action, reward, np.hstack(new_state[:, 0]), new_state[:, 2].any())
                if step % self.learn_every == 0:
                    loss = self.learn_over_replay(stored)
                    losses_of_trial.append(loss)

                states = np.array(new_states[:, :, 0].tolist()).reshape(states.shape)
                if new_states[:, :, 2].any():
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # if trial < self.copy_to_target_at:
            if trial % self.copy_to_target_at == 0:
                self.copy_to_target()
                self.replay_memory.update_mead_std()

            losses.append(np.mean(losses_of_trial))
            total_rewards.append(total_reward)
            # log_metric('mean_episode_loss', np.mean(losses_of_trial), step=trial)
            # log_metric('epsilon', self.epsilon, step=trial)
            # log_metric('queue_size', len(self.replay_memory), step=trial)
            # log_metric('reward', total_reward, step=trial)
            print('Episode', str.zfill(str(trial), 4), '\t|\t',
                  'Reward', f'{total_reward:.4f}', '\t|\t',
                  'Queue', len(self.replay_memory), '\t|\t',
                  'Epsilon', f'{self.epsilon:.4f}', '\t|\t',
                  'Loss', f'{np.mean(losses_of_trial):.4f}', '\t|\t',
                  'AvgReward', f'{np.mean(total_rewards[-5:]) if len(total_rewards) > 5 else 0:.4f}',
                  # 'TargetUpdates', trial // (self.copy_to_target_at // self.parallel_envs), '\t|\t',
                  sep='\t')
        self.model.save('weights.h5')
        json.dump({str(k): str(v) for (k, v) in self.replay_memory.__dict__.items()
                   if k not in ('buffer', ) and not k.startswith('__')},
                  open('params.json', 'w'))
        return self,  np.mean(total_rewards[-10:])