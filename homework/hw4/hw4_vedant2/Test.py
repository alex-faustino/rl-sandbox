# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:30:50 2018

@author: Vedant
"""
import Agents.discreteAgent as DA
import gym
import gridWorld
import numpy as np
from matplotlib.pyplot import plot


env = gym.make('GridWorld-v0')
#env.render('human')
'''
Q, rewards = DA.SARSA(env)
V = np.amax(Q, axis=1)
V = V.reshape(5,5)
pi = np.argmax(Q, axis=1)
pi = pi.reshape(5,5)
plot(rewards);

Q1, rewards1 = DA.QLearn(env)
V1 = np.amax(Q1, axis=1)
V1 = V1.reshape(5,5)
pi1 = np.argmax(Q1, axis=1)
pi1 = pi1.reshape(5,5)
plot(rewards1);
'''
Q2, rewards2 = DA.Reinforce(env)
V2 = np.amax(Q2, axis=1)
V2 = V2.reshape(5,5)
pi2 = np.argmax(Q2, axis=1)
pi2 = pi2.reshape(5,5)
plot(rewards2);