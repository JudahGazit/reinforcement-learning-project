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
        self.q1 = self.create_model()
        self.q1_target = self.create_model()
        self.q2 = self.create_model()
        self.q2_target = self.create_model()
        self.copy_to_target()

    def create_model(self):
        number_of_features = self.num_features * self.batch_frames
        X_input = Input(number_of_features)
        X = X_input
        X = Dense(512, activation='relu')(X)
        X = Dense(512, activation='relu')(X)
        X = Dense(256, activation='relu')(X)
        advantage = Dense(self.num_actions)(X)
        value = Dense(1)(X)
        X = value + (advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True))
        model = keras.Model(inputs=X_input, outputs=X)
        model.compile(loss="mse",
                      optimizer=keras.optimizer_v2.rmsprop.RMSprop(learning_rate=self.learning_rate, clipvalue=1))
        return model

    def copy_to_target(self):
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

    def _create_targets(self, state, action, reward, next_state, done):
        targets_1 = self.q1.predict(state, batch_size=len(state))
        targets_2 = self.q2.predict(state, batch_size=len(state))
        next_action = self.q1.predict(next_state, batch_size=len(state)).argmax(1)
        Q_future_1 = self.q1_target.predict(next_state, batch_size=len(state))[np.arange(len(targets_1)), next_action]
        Q_future_2 = self.q2_target.predict(next_state, batch_size=len(state))[np.arange(len(targets_2)), next_action]
        discounted_rewards = reward + (1 - done) * np.minimum(Q_future_1, Q_future_2) * self.gamma
        td_error = np.abs(targets_1[np.arange(len(targets_1)), action] - discounted_rewards)
        targets_1[np.arange(len(targets_1)), action] = discounted_rewards
        targets_2[np.arange(len(targets_2)), action] = discounted_rewards
        return targets_1, targets_2, td_error.mean()

    def predict(self, states):
        return self.q1.predict(states, batch_size=len(states))

    def fit(self, state, action, reward, next_state, is_done):
        targets_1, targets_2, td_error = self._create_targets(state, action, reward, next_state, is_done)
        self.q1.fit(state, targets_1, epochs=1, verbose=False, batch_size=len(state), shuffle=False)
        self.q2.fit(state, targets_2, epochs=1, verbose=False, batch_size=len(state), shuffle=False)
        return td_error

    def save(self, name):
        self.q1.save(f'{name}.h5')

    def load(self, name):
        self.q1 = keras.models.load_model(f'{name}.h5')
        self.q2 = keras.models.load_model(f'{name}.h5')