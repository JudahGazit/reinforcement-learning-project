import random

import numpy as np

class ReplayMemory:
    def __init__(self, max_size, num_actions):
        self.buffer = [None] * max_size
        self.td_error = np.empty(max_size)
        self.td_error_sum = 0
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.mean = 0
        self.std = 1
        self.std_reward = 1
        self.epsilon = 1e-12
        self.action_count = [0] * num_actions
        self.alpha = 0.8
        self.beta = 0.3
        self.beta_increment_per_sampling = 5e-5

    def append(self, obj, err):
        if self.index < self.size:
            self.action_count[self.buffer[self.index][1]] -= 1
            self.td_error_sum -= self.td_error[self.index]
        self.action_count[obj[1]] += 1
        self.buffer[self.index] = obj
        self.td_error[self.index] = err ** self.alpha
        self.td_error_sum += err ** self.alpha
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        p = self.td_error[:self.size] / self.td_error[:self.size].sum()
        indices = np.random.choice(self.size, batch_size - 1, False, p=p).tolist() + [self.index - 1]
        w = (self.size * p[indices]) ** (- self.beta)
        self.beta = min([self.beta + self.beta_increment_per_sampling, 1])
        return indices, np.array([self.buffer[i] for i in indices]), w

    def update(self, indices, err):
        self.td_error_sum -= self.td_error[indices].sum()
        self.td_error[indices] = err ** self.alpha
        self.td_error_sum += self.td_error[indices].sum()

    def update_mead_std(self):
        states = np.array(self.buffer[:self.size])[:, 0]
        rewards = np.array(self.buffer[:self.size])[:, 2]
        self.mean = np.mean(states, 0)
        self.std = np.std(states, 0)
        self.std_reward = np.std(rewards)

    def normalize_state(self, states):
        normed = (states - self.mean) / (self.std + self.epsilon)
        return normed

    def normalize_reward(self, reward):
        return reward / (self.std_reward + self.epsilon)

    def __len__(self):
        return self.size