# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 01:59:46 2018

@author: Vedant
"""
import numpy as np

import matplotlib.pyplot as plt
import torch
from matplotlib import animation, rc
import SASA_mujoco_env
import PPO_par as ppo

gamma = 0.999
lamb = 0.95
number_of_actors =5
number_of_iterations = 10

number_of_epochs = 50
minibatch_size = 1000
logstd_initial = -1 #-0.7
logstd_final = -2 # -1.6
epsilon = 0.2
use_multiprocess = False

horizon = int(1e4*30) #secs #steps

Noise = None
render = False


env = SASA_mujoco_env.SASAmujocoEnv()
agent = ppo.PPOAgent(env)


res = agent.train(
    'SASA',
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

T = horizon
a_actual = np.zeros((T, env.action_dim))
env.reset()
s = np.zeros((T, env.observation_dim))
r = np.zeros(T)
a = np.zeros((T, env.action_dim))
a_actual = np.zeros((T, env.action_dim))
time = np.zeros(T)
s[0,:] = env.x
for t in range(1,T):
    a[t] = agent.action_greedy(s[t-1,:])
    s[t,:], r[t], _ , _= env.step(a[t])
    a_actual[t] = env.a
    
    
plt.plot(res['rewards'])
plt.xlabel('iteration')
plt.ylabel('reward');
plt.show()

plt.plot(a_actual)
plt.ylabel('Action')
plt.xlabel('Time')
plt.legend(('action_x', 'action_y', 'action_z'))
plt.show()

plt.ylabel('Quaternion')
plt.xlabel('Time')
plt.legend()
plt.plot(s[:,0:4], label='x')
plt.show()

plt.ylabel('Angular Velocity')
plt.xlabel('Time')
plt.legend()
plt.plot(s[:,0:3], label='x')
plt.show()