# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:37:50 2018

@author: Vedant
"""

import gym,sys
import acrobot

import Agents.DQN_Agent as DQNA

env = gym.make('vedant_acrobot-v0')


#Q, rewards = 
r = DQNA.DQN_Agent(env,BATCH_SIZE = 150,GAMMA = 0.999,TARGET_UPDATE = 50,initial_epsilon = 1, 
              final_epsilon = 0.1,total_episodes = 5000, annealing_period = None,max_steps = 100)
