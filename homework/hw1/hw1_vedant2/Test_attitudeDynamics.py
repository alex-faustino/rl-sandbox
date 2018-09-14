# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 02:14:59 2018

@author: Vedant
"""

import gym,sys
import attitudeDynamics
import numpy as np

env = gym.make('attitudeDynamics-v0')



def run_episode(env, parameters):
    action_list = []
    reward_list= []
    observation_list= []
    observation = env.reset()
    totalreward = 0
    for i in range(500):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalreward += reward
        #action_list = np.append(action_list, (action))
        reward_list= np.append(reward_list, reward)
        #observation_list= np.append(observation_list, (observation))
        action_list.append(action)
        observation_list.append(observation)
    return totalreward, action_list, reward_list, observation_list
parameters = np.random.rand(4) * 2 - 1
reward, action_list, reward_list, observation_list = run_episode(env,parameters)
env.close()