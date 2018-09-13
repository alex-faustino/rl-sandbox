import gym
import random
import numpy as np
import time
from grid_world import GridWorldEnv
from drop import DropEnv

# env = gym.make('GridWorld-v0')
# env = env.unwrapped
# env.random(True)
# observation = env.reset()
# action_map = {0:'left', 1:'up', 2:'right', 3:'down'}
# print('initial state =', observation)
# for t in range(40):
#     env.render('human')
#     action = env.action()
#     observation, reward, done, info = env.step(action)
#     time.sleep(0.5)
#     print('time = {}, action = {}, goto = {}, reward = {}'.format(
#         t, action_map[action], observation, reward))

env = DropEnv()
env = env.unwrapped
observation = env._reset()
print('initial state =', observation)
for t in range(40):
    env._render('human')
    action = env.new_drop(3)
    observation, reward, done, info = env._step(action)
    time.sleep(0.5)
    print('time = {}, new_drop = {}, goto = {}, reward = {}'.format(
        t, action, observation, reward))
