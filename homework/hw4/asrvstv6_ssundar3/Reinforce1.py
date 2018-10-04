#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:56:20 2018

@author: AmberS
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import gym_envs
import random
from random import choices

env = gym.make('grid_world-v0')
M = env.action_space.n
N1 = env.state_space.n
breakout = True
N = N1*N1
k = 0
d = {}
for i in range(N1):
    for j in range(N1):
        d.update([((i,j),k)])
        k+=1

theta = np.zeros((M,N))
rew = np.ones((M,N))

alpha = 0.00005
ep_max = 1000
epsilon = 0.9
t_steps = 100
p = np.zeros((M,N))
dist = [epsilon, 1-epsilon]
choice = [1,2]

def softmax(s):
    
    #p_theta = np.exp(np.multiply(theta[:,s],rew[:,s]))
    p_theta = np.exp(theta[:,s])
    #print(np.linalg.norm(theta))
    return p_theta/np.sum(p_theta), np.argmax(p_theta/np.sum(p_theta))

count= 0

def softmax_complete():
    P_Theta = np.zeros((M,N))
    for i in range(N):
        p_theta = np.exp(np.multiply(theta[:,i],rew[:,i]))
        p_theta = p_theta/np.sum(p_theta)
        P_Theta[:,i] = p_theta
        #print(np.multiply(theta[:,s],rew[:,s]))
        #print(rew[:,s])
    print('Finally:', P_Theta)
    #tmp = np.argmax(P_Theta,axis=0)
    tmp_star = np.zeros((N1,N1))
    for i in range(N1):
        for j in range(N1):
            s = d[(i,j)]
            f1 = np.argmax(P_Theta[:,s])
            tmp_star[4-i,j] = f1
    print(tmp_star)
    
    
while breakout:
    count += 1
    
    tt = theta
    for i in range(ep_max):
        
        grd = np.zeros((t_steps,M,N))
        env.reset()
        s1,s2 = env.state
        s_arr = np.zeros(t_steps+1, dtype=np.int8) # to store the sequence of states in T steps
        a_arr = np.zeros(t_steps+1, dtype=np.int8) # to store the sequence of actions in T steps
        r_arr = np.zeros(t_steps+1) # to store the sequence of rewards in T steps.
        s_arr[0] = d[(s1,s2)]
          
        for t in range(t_steps):
            ep_choice = random.choices(choice, dist)
            
            if ep_choice == [2]:
                tmp, a_arr[t] = softmax(s_arr[t]) 
                
            else:
                a_arr[t] = np.random.randint(0,M) 
                
            
            obs, reward, done, info = env.step(a_arr[t])
            
            s1,s2 = obs
            s_arr[t+1] = d[(s1,s2)]
            r_arr[t+1] = reward
            phi = np.zeros((M,N))
           
            phi[a_arr[t],s_arr[t]] = 1
            grd[t,:,:] += phi 
            p_theta, a_arr[t] = softmax(s_arr[t])
            
            for ac in range(M):
                phi = np.zeros((M,N))
                phi[ac,s_arr[t]] = 1
                grd[t,:,:] -= p_theta[ac]*phi
            
        for t in range(t_steps):
            v_t = 0
            
            for t1 in range(t+1, t_steps+1):
                v_t += r_arr[t1]
                
            theta = theta + alpha*(grd[t,:,:])*v_t
        
    print(np.linalg.norm(theta-tt))  
    epsilon = epsilon * 0.9        
        #print('Gradient: ',grd[i,:,:])
        #print(' Reward:', R[i])
        #ep_sum += (1/ep_max)*R[i]*grd[i,:,:] 
     
    print('Count is: ', count)
    #theta = theta_old + alpha*ep_sum
    #print(theta)
    #print(ep_sum)
    #print(np.linalg.norm(theta-theta_old))
    
    if count > 1000:
        softmax_complete()
        breakout = False
    
    #softmax_complete()
      
