# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 01:59:46 2018

@author: Vedant
"""
import numpy as np

import matplotlib.pyplot as plt
import torch
from matplotlib import animation, rc
import sat_mujocoenv
import PPO_par as ppo

env = sat_mujocoenv.Sat_mujocoEnv(maxabs_torque=1, 
                     target_state = np.array([0,0,0,1,0,0,0]), w_mag = 2.5e-3 ,
                     w_tumble = None, Noise = None,visualize = False)
agent = ppo.PPOAgent(env)

gamma = 0.99
lamb = 0.95
number_of_actors = 50
number_of_iterations = 1000
horizon = 200
number_of_epochs = 100
minibatch_size = 100
logstd_initial = -1 #-0.7
logstd_final = -2 # -1.6
epsilon = 0.2
use_multiprocess = False
res = agent.train(
    'Sat',
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

plt.plot(res['rewards'])
plt.xlabel('iteration')
plt.ylabel('reward');
plt.show()

plt.plot(res['losses'], label='L')
plt.plot(res['losses_clip'], label='L_clip')
plt.plot(res['losses_V'], label='L_V')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend();
plt.show()

plt.plot(res['losses_V'], label='L_V')
plt.xlabel('iteration')
plt.ylabel('loss_V');
plt.show()

plt.plot(res['stds'])
plt.xlabel('iteration')
plt.gca().set_ylim(bottom=0)
plt.ylabel('standard deviation');
plt.show()

plt.plot(res['times_sample'], label='sample')
plt.plot(res['times_opt'], label='opt')
plt.gca().set_ylim(bottom=0)
plt.legend();
plt.xlabel('iteration')
plt.ylabel('time per iteration / seconds');
plt.show()
