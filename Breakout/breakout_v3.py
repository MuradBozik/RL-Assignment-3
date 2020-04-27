# -*- coding: utf-8 -*-
"""breakout_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZtFnS4SYYOufkOPQSDdnkw4alQgFaC86
"""

from collections import deque
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import errno
import uuid
import gym
import cv2
import os

# from google.colab import drive
# drive.mount('/content/drive')


# Global PARAMETERS
episodes = 5000
epsilon = 1.0
epsilon_min = 0.1
discount_rate = 0.95
learning_rate = 0.01
batch_size = 32
update_interval = 10
memory_size = 2000

# model_path = './drive/My Drive/Colab Notebooks/models/breakoutDQN.h5'
model_path = '../models/breakoutDQN.h5'
config_path = './trainings/{}/config'
summary_path = './trainings/{}/detailedSummary.txt'


class QAgent:
    def __init__(self, environment):
        self.action_size = environment.action_space.n
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_interval = update_interval  # how many steps for update between the target model and model
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.model = self.load_model()
        self.target_model = keras.models.clone_model(self.model)
        self.max_reward = 0

    def load_model(self):
        global model_path
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                # We are decreasing epsilon because model does not need to be eager to explore
                # self.epsilon = 0.01
                return model
            except IOError:
                print("Error happened during loading the model! New model is being created!")
                return self.create_model()
        else:
            print("Model cannot found! New model is being created!")
            return self.create_model()

    @staticmethod
    def pre_processing(frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        # Turning to graysclae
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t/255

    def create_model(self):
        """Returns: created cnn model
        We created cnn model, because we are dealing with images (observations)"""
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(32, kernel_size=[8, 8], strides=[4, 4], input_shape=[84, 84, 1],
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=0.01,
                                                                                            distribution='normal'),
                                      padding='same', name='conv1'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=[4, 4], strides=[2, 2], padding='same', name='conv2',
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=0.01,
                                                                                            distribution='normal'),
                                      ))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=[3, 3], strides=[1, 1], padding='same', name='conv3',
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=0.01,
                                                                                            distribution='normal')))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, name='fc1'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(self.action_size, name='fc2'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        # print(model.summary())
        return model

    def get_random_action(self):
        """Returns a random action,
        This will be used not to stuck the same actions and continue to explore state space"""
        return np.random.choice(range(self.action_size))

    def get_best_action(self, observation):
        """Returns: the action with max Q value for a particular state"""
        input_ = self.pre_processing(observation)
        output = self.model.predict(input_[np.newaxis])
        return np.argmax(output[0])

    def get_action(self, state_):
        # Here is epsilon greedy policy
        return self.get_random_action() if np.random.rand() < self.epsilon else self.get_best_action(state_)

    def sample_experiences(self):
        indices = np.random.randint(len(self.memory), size=self.batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, next_states, rewards, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        processed_states = np.array([self.pre_processing(state_) for state_ in states])
        processed_next_states = np.array([self.pre_processing(next_state_) for next_state_ in next_states])

        return processed_states, actions, processed_next_states, rewards, dones, batch

    def train(self, eps):
        global epsilon_min

        # This step is to wait until we get enough state, controls by memory size
        if len(self.memory) < self.memory_size:
            return

        if eps % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

        # Sample batch-sized memory
        states_, actions_, next_states_, rewards_, dones_, experiences_ = self.sample_experiences()

        Q_table = self.model.predict(states_)
        Q_next = self.target_model.predict(next_states_)
        # Update Q_table
        Q_table[np.arange(self.batch_size), actions_] = (
                rewards_ + (1 - dones_) * self.discount_rate * np.max(Q_next, axis=1))

        self.model.fit(states_, Q_table, verbose=0)
        # Update epsilon
        agent.epsilon = agent.epsilon * (1 - 1e-3) if agent.epsilon > epsilon_min else epsilon_min

    def store_memory(self, state_, action_, next_state_, reward_, done_):
        self.memory.append((state_, action_, next_state_, reward_, done_))


def save_configs(detail_lines):
    """
    Purpose of this function is to store the used parameters for this training session
    """
    configurations = {
        'episodes': episodes,
        'epsilon': epsilon,
        'epsilon_min': epsilon_min,
        'discount_rate': discount_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'update_interval': update_interval,
        'memory_size': memory_size
    }

    training_id = str(uuid.uuid4())
    config_filename = config_path.format(training_id)
    if not os.path.exists(os.path.dirname(config_filename)):
        try:
            os.makedirs(os.path.dirname(config_filename))
        except OSError as e:  # Guard against race condition
            if e.errno != errno.EEXIST:
                raise

    with open(config_filename, "wb") as f:
        pickle.dump(configurations, f)

    detailedSummary_filename = summary_path.format(training_id)
    with open(detailedSummary_filename, "w") as f:
        f.writelines(detail_lines)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    env = gym.make('Breakout-v0')
    agent = QAgent(environment=env)

    detailedSummary = []
    hit_rewards = list()
    for episode in range(episodes):
        state = env.reset()
        # print("state.shape: ", state.shape)
        current_reward = 0
        while True:
            action = agent.get_action(state)
            next_s, reward, done, info = env.step(action)
            agent.store_memory(state, action, next_s, reward, done)
            agent.train(episode)
            current_reward += reward
            state = next_s
            if done:
                hit_rewards.append(current_reward)
                line = f'Episode:{episode}, eps_reward:{current_reward}, max_hit_reward:{max(hit_rewards)}, ' \
                       f'agent.max_reward: {agent.max_reward}, epsilon: {agent.epsilon}'
                detailedSummary.append(line)
                print(line)
                break

        if np.max(hit_rewards) > agent.max_reward:
            agent.max_reward = max(hit_rewards)
            if not os.path.exists(model_path):
                try:
                    os.makedirs(os.path.dirname(model_path))
                except OSError as exc:  # Guard against race condition
                    pass
            agent.model.save(model_path)
            print("Model saved!")

    save_configs(detailedSummary)
    env.close()
