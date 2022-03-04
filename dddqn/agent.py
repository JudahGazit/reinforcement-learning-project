import itertools
import json
import logging
import random
import warnings

import gym
import numpy as np

from dddqn.dddqn import DDDQN
from dddqn.replay_memory import ReplayMemory

warnings.filterwarnings("ignore")

logger = logging.getLogger('Agent')


class Agent:
    def __init__(self, env, batch_frames, action_step_size=3, copy_to_target_at=10, learning_rate=0.00005478):
        super().__init__()
        self.env_name = env
        self.env = gym.make(self.env_name).env

        self.batch_frames = batch_frames
        self.copy_to_target_at = copy_to_target_at
        self.learn_every = 2
        self.minimum_states = 5000
        self.batch_size = 128
        self.epsilon = 1
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.99
        self.replay_memory_size = 1_000_000
        self.action_step_size = action_step_size
        self.learning_rate = learning_rate

        for k, v in self.__dict__.items():
            if k != 'env':
                logger.info(f'Setting param: {k} = {v}')

        self.action_space = self._make_action_space(action_step_size)
        self.replay_memory = ReplayMemory(self.replay_memory_size, len(self.action_space))
        self.model = DDDQN(self.env.observation_space.shape[0], len(self.action_space), self.batch_frames,
                           self.learning_rate, self.action_step_size)

    def _make_action_space(self, action_step_size):
        number_of_axes = self.env.action_space.shape[0]
        actions = itertools.product(*[np.linspace(-1, 1, action_step_size) for _ in range(number_of_axes)])
        actions = np.array(list(actions))
        actions = actions[(np.sum(actions != 0, 1) <= 4) & (np.sum(actions != 0, 1) > 0)]
        return actions

    def _sample_from_replay(self, stored):
        if stored is not None:
            return np.concatenate([self.replay_memory.sample(self.batch_size - 1), [stored]])
        return np.array(self.replay_memory.sample(self.batch_size))

    def _train_episode(self, episode_number, episode_length):
        state = self.reset()
        total_reward = 0
        losses_of_trial = []
        for step in range(episode_length):
            state, reward, is_done, loss = self._step(state, step)
            total_reward += reward
            losses_of_trial.append(loss)
            if is_done.any():
                break
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        loss_of_trial = np.mean(losses_of_trial)
        if episode_number % self.copy_to_target_at == 0:
            self.model.copy_to_target()
            self.replay_memory.update_mead_std()
        self._log_episode(episode_number, loss_of_trial, total_reward)
        return total_reward, loss_of_trial

    def _step(self, state, step_number, train=True):
        action = self.act(state, stochastic=train)
        next_state = np.array([self.env.step(self.action_space[action]) for _ in range(self.batch_frames)])
        next_state, reward, is_done, info = [np.hstack(next_state[:, i]) for i in range(next_state.shape[1])]
        reward_clipped = np.clip(reward.sum(), -10, 1)
        if train:
            stored = self.remember(state, action, reward_clipped, next_state, is_done.any() and reward_clipped < -5)
            loss = 0
            if step_number > 0 and step_number % self.learn_every == 0:
                loss = self.learn_over_replay(stored)
            return next_state, np.max([reward.sum(), -100]), is_done.any(), loss
        else:
            return next_state, np.max([reward.sum(), -100]), is_done.any()

    def _log_episode(self, episode_number, loss_of_trial, total_reward):
        logger.info(
            '\t|\t'.join([f'Episode {str.zfill(str(episode_number), 4)}',
                          f'Reward {total_reward:.4f}',
                          f'Queue {len(self.replay_memory)}',
                          f'Epsilon {self.epsilon:.4f}',
                          f'Loss {loss_of_trial:.4f}'])
        )

    def learn_over_replay(self, stored=None):
        if len(self.replay_memory) >= self.minimum_states:
            samples = self._sample_from_replay(stored)
            state, action, rewards, next_state, is_done = [samples[:, i] for i in range(samples.shape[1])]
            state = self.replay_memory.normalize_state(np.vstack(state))
            next_state = self.replay_memory.normalize_state(np.vstack(next_state))
            rewards = self.replay_memory.normalize_reward(rewards)
            loss = self.model.fit(state, action.astype(int), rewards, next_state, is_done)
            return loss
        else:
            self.replay_memory.update_mead_std()
            self.epsilon = 1
        return 0

    def remember(self, state, action, reward, new_state, done):
        payload = [state, action, reward, new_state, done]
        self.replay_memory.append(payload)
        return payload

    def act(self, state, stochastic=True):
        action = np.argmax(self.model.predict(self.replay_memory.normalize_state(state.reshape(1, -1))), 1)[0]
        if (np.random.random() < self.epsilon) and stochastic:
            if self.replay_memory.size:
                weights = np.array([1 - a / self.replay_memory.size for a in self.replay_memory.action_count])
                action = np.random.choice(len(self.action_space), p=weights / np.sum(weights))
            else:
                action = random.choices(range(len(self.action_space)))[0]
        return action

    def train(self, episodes=3000, episode_length=2000, finish_after=10):
        losses = []
        total_rewards = []
        for trial in range(episodes):
            to_stop = len(total_rewards) > finish_after and np.mean(total_rewards[-finish_after:]) > 300
            if not to_stop:
                total_reward, loss = self._train_episode(trial, episode_length)
                total_rewards.append(total_reward)
                losses.append(loss)

        return self

    def play(self, length=2000, render=True):
        state = self.reset()
        states = [state]
        frames = []
        rewards = []
        for i in range(length):
            state, reward, is_done = self._step(state, i, train=False)
            states.append(state)
            rewards.append(reward)
            if render:
                frames.append(self.env.render(mode='rgb_array'))
            if is_done:
                break
        return (rewards, frames) if render else rewards

    def reset(self):
        state = self.env.reset()
        state = np.hstack([state for _ in range(self.batch_frames)])
        return state

    def save(self, name):
        self.model.save(name)
        json.dump({str(k): str(v) for (k, v) in self.replay_memory.__dict__.items()
                   if k not in ('buffer',) and not k.startswith('__')},
                  open(f'{name}.json', 'w'))

    def load(self, name):
        self.model.load(name)
        j = json.load(open(f'{name}.json'))
        for k, v in j.items():
            if '[' in v:
                v = np.fromstring(v[1:-1].replace(',', ''), sep=' ')
            elif float(v).is_integer():
                v = int(v)
            else:
                v = float(v)
            setattr(self.replay_memory, k, v)
        return self
