# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:37:50 2018

@author: Vedant
"""

import gym,sys
import acrobot
import numpy as np

env = gym.make('vedant_acrobot-v0')
#env = gym.make('CartPole-v0')
num_episodes = 200
max_timestep = 1000

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for i in range(5000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalreward += reward
    return totalreward
parameters = np.random.rand(4) * 2 - 1
reward = run_episode(env,parameters)
env.close()