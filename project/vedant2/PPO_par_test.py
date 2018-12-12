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

gamma = 0.99
lamb = 0.95
number_of_actors =10
number_of_iterations = 100
horizon = 1000
number_of_epochs = 50
minibatch_size = 500
logstd_initial = -1 #-0.7
logstd_final = -2 # -1.6
epsilon = 0.1
use_multiprocess = False

horizon = 1000*2 #steps
maxabs_torque=1.0e-2
dt = 10/2
target_state = np.array([1,0,0,0,0,0,0]) # [q_0,q_1,q_2,q_3,w_0,w_1,w_2]
w_mag = 4e-2
w_tumble = 8e-2
Noise = None
render = False


env = sat_mujocoenv.Sat_mujocoEnv(horizon,maxabs_torque, dt ,
                     target_state , w_mag  ,
                     w_tumble , Noise ,render )
agent = ppo.PPOAgent(env)


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
plt.plot(s[:,4:], label='x')
plt.show()