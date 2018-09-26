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
DQNA.DQN_Agent(env)
