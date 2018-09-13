import gym
import random
import numpy as np
import time
from grid_world import GridWorldEnv
import signal
import sys

episode_length = 40
action_map = {0:'left', 1:'up', 2:'right', 3:'down'}
env = GridWorldEnv('SARSA', 'hard')
env = env.unwrapped

def sigint_handler(signum, frame):
    env.draw()
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

while True:
    observation = env._reset()
    print('initial state =', observation)
    for t in range(episode_length):
        env._render('human')
        action = env.action()
        observation, reward, done, info = env._step(action)
        # time.sleep(0.1)
        print('time = {}, action = {}, goto = {}, reward = {}'.format(
            t, action_map[action], observation, reward))
