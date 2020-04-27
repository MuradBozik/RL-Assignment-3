import time
import gym
from Breakout.breakout_v6 import QAgent

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
model_path = 'models/breakoutDQN_v6.h5'
config_path = './trainings/{}/config'
summary_path = './trainings/{}/detailedSummary.txt'

env = gym.make('Breakout-v0')
state = env.reset()
rewards = 0
agent = QAgent(environment=env)

while True:
    env.render()
    time.sleep(0.01)
    action = agent.get_action(state)
    state, reward, done, _ = env.step(action)
    rewards += reward
    if done:
        print('rewards:', rewards)
        break
env.close()
