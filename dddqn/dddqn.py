import copy

import keras.models
import keras.optimizer_v2
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Dense


class DDDQN:
    def __init__(self, num_features, num_actions, batch_frames, learning_rate, action_step_size, gamma=0.99):
        self.gamma = gamma
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.batch_frames = batch_frames
        self.action_step_size = action_step_size
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.copy_to_target()

    def create_model(self):
        number_of_features = self.num_features * self.batch_frames
        X_input = Input(number_of_features)
        X = X_input
        X = Dense(512, activation='relu')(X)
        X = Dense(512, activation='relu')(X)
        X = Dense(256, activation='relu')(X)
        advantage = Dense(4 * 3)(X)
        advantage = tf.reshape(advantage, (-1, 4, 3))
        value = Dense(1)(X)
        X = tf.expand_dims(value, 1) + (advantage - tf.math.reduce_mean(advantage, axis=2, keepdims=True))
        model = keras.Model(inputs=X_input, outputs=X)
        model.compile(loss="mse",
                      optimizer=keras.optimizer_v2.rmsprop.RMSprop(learning_rate=self.learning_rate, clipvalue=1))
        return model

    def copy_to_target(self):
        self.target_model.set_weights(copy.deepcopy(self.model.get_weights()))

    def _create_targets(self, state, action, reward, next_state, done):
        targets = self.model.predict(state, batch_size=len(state))
        next_action = self.model.predict(next_state, batch_size=len(state)).argmax(2) - 1
        target_Q = self.target_model.predict(next_state, batch_size=len(state))
        Q_future = np.array([target_Q[i, np.arange(4), next_action[i] + 1] for i in range(len(action))])
        discounted_rewards = reward[:, None] + (1 - done)[:, None] * Q_future * self.gamma
        td_error = np.abs(np.array([targets[i, np.arange(4), action[i] + 1] for i in range(len(action))]).sum(1) - discounted_rewards.sum(1))
        for i in range(len(action)):
            targets[i, np.arange(4), action[i] + 1] = discounted_rewards[i]
        return targets, td_error.mean()

    def predict(self, states):
        return self.model.predict(states, batch_size=len(states))

    def fit(self, state, action, reward, next_state, is_done):
        targets, td_error = self._create_targets(state, action, reward, next_state, is_done)
        self.model.fit(state, targets, epochs=1, verbose=False, batch_size=len(state), shuffle=False)
        return td_error

    def save(self, name):
        self.model.save(f'{name}.h5')

    def load(self, name):
        self.model = keras.models.load_model(f'{name}.h5')
        self.target_model = keras.models.load_model(f'{name}.h5')