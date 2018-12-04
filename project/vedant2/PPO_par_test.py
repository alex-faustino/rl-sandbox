# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 01:59:46 2018

@author: Vedant
"""

import gym,sys
import attitudeDynamics
import numpy as np

import matplotlib.pyplot as plt
import torch
from matplotlib import animation, rc

import PPO_par as ppo

env = gym.make('attitudeDynamics-v0')
agent = ppo.PPOAgent(env)

gamma = 0.99
lamb = 0.95
number_of_actors = 50
number_of_iterations = 1000
horizon = 20
number_of_epochs = 100
minibatch_size = 100
logstd_initial = -1 #-0.7
logstd_final = -2 # -1.6
epsilon = 0.2
use_multiprocess = True
res = agent.train(
    'pend',
    gamma,
    lamb,
    number_of_actors,
    number_of_iterations,
    horizon,
    number_of_epochs,
    minibatch_size,
    logstd_initial,
    logstd_final,
    epsilon,
    use_multiprocess,
)