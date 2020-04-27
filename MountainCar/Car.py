# dqn.py
# https://geektutu.com
from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers,utils
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# init model
model = models.Sequential()
model.add(layers.Dense(100, input_dim=2, activation='relu'))
model.add(layers.Dense(3, activation="linear"))
model.compile(loss='mean_squared_error',
                optimizer=optimizers.Adam(0.001))
utils.plot_model(model, to_file='model.png')

target_model = models.Sequential()
target_model.add(layers.Dense(100, input_dim=2, activation='relu'))
target_model.add(layers.Dense(3, activation="linear"))
target_model.compile(loss='mean_squared_error',
                optimizer=optimizers.Adam(0.001))

# init model      
car_step = 0
steps_for_update = 200  # how many steps for update between the target model and model
the_memory_size = 2000  
memory = deque(maxlen=the_memory_size)

# init env
env = gym.make('MountainCar-v0')
episodes = 1000
hit_rewards = [] 
try_rewards = []

def store_memory(state, action, next_state, reward):
    global car_step,steps_for_update,the_memory_size,memory,model,target_model
    # give more award for initialize state.
    try_rewards = []
    try_rewards.append(3)
    if next_state[0] >= 0.4:
        reward += try_rewards[0]
    memory.append((state, action, next_state, reward))

def train(batch_size=64, learning_rate=1):
    global car_step,steps_for_update,the_memory_size,memory,model,target_model
    if len(memory) < the_memory_size:
            return
    car_step += 1
    # When we have done steps_for_updates we need to assing the value to the target_model
    if car_step % steps_for_update == 0:
        target_model.set_weights(model.get_weights())
    memory_batch = random.sample(memory, batch_size)
    state_for_train = np.array([memory[0] for memory in memory_batch])
    next_state_for_train = np.array([memory[2] for memory in memory_batch])

    Q_table = model.predict(state_for_train)
    Q_next = target_model.predict(next_state_for_train)

    for i, s_memory in enumerate(memory_batch):
        _, a, _, reward = s_memory
        Q_table[i][a] = (1 - learning_rate) * Q_table[i][a] + learning_rate * (reward + 0.95 * np.amax(Q_next[i]))
        # 0.95 is lambda

    model.fit(state_for_train, Q_table, verbose=0)


for i in range(episodes):
    state = env.reset()
    current_reward = 0
    while True:
        try_rewards = []
        try_rewards.append(0.002)
        if np.random.uniform() < 0.3 - car_step * try_rewards[0]:
            action =  np.random.choice([0, 1, 2])
        action = np.argmax(model.predict(np.array([state]))[0])
        next_s, reward, done, _ = env.step(action)
        store_memory(state, action, next_s, reward)
        train()
        current_reward += reward
        state = next_s
        if done:
            hit_rewards.append(current_reward)
            print('episode:', i, 'current_reward:', current_reward, 'max:', max(hit_rewards))
            break
    if np.mean(hit_rewards[-10:]) > -160:
        model.save('DQN.h5')
        break
env.close()

import matplotlib.pyplot as plt

plt.plot(hit_rewards)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()