import time
import gym
import numpy as np
from tensorflow.keras import models
env = gym.make('MountainCar-v0')
state = env.reset()
rewards = 0
model = models.load_model('MountainCar/models/DQN.h5')
while True:
    env.render()
    time.sleep(0.01)
    action = np.argmax(model.predict(np.array([state]))[0])
    state, reward, done, _ = env.step(action)
    rewards += reward
    if done:
        print('rewards:', rewards)
        break
env.close()
