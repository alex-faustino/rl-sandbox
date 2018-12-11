# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:23:02 2018

@author: vedant2
"""

import sat_mujocoenv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2 as algorithm

import numpy as np
import matplotlib.pyplot as plt
import time

a = time.time()
horizon = 1500 #steps
maxabs_torque=2.50e-3
dt = 5
target_state = np.array([1,0,0,0,0,0,0]) # [q_0,q_1,q_2,q_3,w_0,w_1,w_2]
w_mag = 4e-2
w_tumble = 1e-1
Noise = None
render = False

env = sat_mujocoenv.Sat_mujocoEnv(horizon,maxabs_torque, dt ,
                     target_state , w_mag  ,
                     w_tumble , Noise ,render )
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = algorithm(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=1000000)
b = time.time()
print("Done Training!")
print('It took : ', b-a,' secs')
Final_run_nstep = 1000
rewards = np.ones([Final_run_nstep])
obs_list = np.zeros([Final_run_nstep+1,env.envs[0].observation_dim])
action_list = np.zeros([Final_run_nstep+1,env.envs[0].action_dim])

obs = env.reset()
for i in range(Final_run_nstep):
    action, _states = model.predict(obs)
    obs, rewards[i], dones, info = env.step(action)
    obs_list[i+1] = obs
    action_list[i] = action
    
    #print(rewards[i])



plt.plot(rewards)
plt.xlabel('iteration')
plt.ylabel('reward');
plt.show()
'''
plt.plot(action_list)
plt.xlabel('iteration')
plt.ylabel('reward');
plt.show()

plt.plot(obs_list[:,1:4])
plt.xlabel('iteration')
plt.ylabel('quaternion');
plt.show()

plt.plot(obs_list[:,4:])
plt.xlabel('iteration')
plt.ylabel('quaternion');
plt.show()
'''