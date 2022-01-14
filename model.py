import numpy as np
import random
import warnings
from collections import deque

import numpy as np
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Sequential

import shelve

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, env, batch_frames, action_step_size=4):
        self.env = env

        self.action_step_size = action_step_size
        self.copy_to_target_at = 32
        self.number_of_replays = 0
        self.minimum_states = 1000
        self.batch_frames = batch_frames
        self.batch_size = 32
        self.gamma = 0.85
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.action_space = self.__make_action_space()

        replay_memory_pickle = shelve.open('replay.pickle', writeback=True)
        replay_memory_pickle['mem'] = deque(maxlen=10_000)
        self.replay_memory = replay_memory_pickle['mem']
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.copy_to_target()


    def __make_action_space(self):
        number_of_axes = self.env.action_space.shape[0]
        # actions = itertools.product(*[(-self.action_step_size, 0, self.action_step_size) for _ in range(number_of_axes)])
        # actions = itertools.product(*[np.linspace(-1, 1, self.action_step_size) for _ in range(number_of_axes)])
        # actions = np.array(list(actions))
        actions = (np.linspace(-1, 1, self.action_step_size) * np.eye(number_of_axes)[:, :, None]).reshape(number_of_axes, -1, ).transpose()
        actions = np.concatenate([actions, np.zeros((1, number_of_axes))])
        print('Number of actions', len(actions))
        return actions

    def create_model(self):
        model = Sequential()
        number_of_features = self.env.observation_space.shape[0] * self.batch_frames
        print('Number of features', number_of_features)
        model.add(Input(number_of_features))
        model.add(BatchNormalization())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(len(self.action_space)))
        model.compile(loss="mean_squared_error", optimizer='adam')
        model.trainable = True
        return model

    def copy_to_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, new_state, done):
        self.replay_memory.append([state, action, reward, new_state, done])

    def act(self, states, stochastic=True):
        greedy_actions = np.argmax(self.model.predict(states), 1)
        actions = np.array([
                            np.random.choice(len(self.action_space)) if (np.random.random() < self.epsilon) and stochastic
                            else greedy_action
                            for greedy_action in greedy_actions
        ])
        return actions

    def learn_over_replay(self):
        if self.number_of_replays % self.copy_to_target_at == 0:
            self.copy_to_target()
        if len(self.replay_memory) >= self.minimum_states:
            self.number_of_replays += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            samples = np.array(random.sample(self.replay_memory, self.batch_size))
            targets = self.target_model.predict(np.vstack(samples[:, 0]))
            Q_future = self.target_model.predict(np.vstack(samples[:, 3])).max(1)
            targets[np.arange(len(targets)), samples[:, 1].astype(int)] = samples[:, 2] + (1 - samples[:, 4]) * Q_future * self.gamma
            loss = self.model.fit(np.vstack(samples[:, 0]), targets, epochs=1, verbose=False, batch_size=self.batch_size, shuffle=False)
            return loss

    def train(self, episodes=200, episode_length=500):
        for trial in range(episodes):
            cur_state = self.env.reset()
            cur_state = np.concatenate([cur_state, np.zeros((self.batch_frames - 1) * len(cur_state))])
            total_reward = 0
            for step in range(episode_length):
                action = self.act(cur_state.reshape(1, -1))[0]
                batch_states = np.array([self.env.step(self.action_space[action]) for _ in range(self.batch_frames)])
                new_state, reward, done, _ = np.concatenate(batch_states[:, 0]), *batch_states[-1, 1:]

                reward = np.max([reward, -10])
                total_reward += reward
                self.remember(cur_state, action, reward, new_state, done)
                self.learn_over_replay()

                cur_state = new_state
                if done:
                    break
            print('Episode', trial, 'Reward', total_reward, 'Queue', len(self.replay_memory), 'Epsilon', self.epsilon)
        return self