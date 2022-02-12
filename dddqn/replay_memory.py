import random

import numpy as np

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.mean = 0
        self.std = 1
        self.std_reward = 1
        self.epsilon = 1e-12

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
        self.std_reward = np.std(rewards)

    def normalize_state(self, states):
        normed = (states - self.mean) / (self.std + self.epsilon)
        return normed

    def normalize_reward(self, reward):
        return reward / (self.std_reward + self.epsilon)

    def __len__(self):
        return self.size