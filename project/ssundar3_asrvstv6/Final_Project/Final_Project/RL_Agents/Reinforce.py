# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 00:40:01 2018

@author: sreen
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from random import choices
import matplotlib.pyplot as plt

class ReinforceAgent(object):
    
    def __init__(self,env):
        self.env = env


    def train(self,alpha,ep_max,epsilon,t_steps,episodes):
        #env = gym.make('grid_world-v0')
        M = self.env.action_space.n
        N = self.env.observation_space.n

        theta = np.zeros((M,N))
        #choice_a = [0,1,2,3]
        choice_a = [i for i in range(M)]
        
        rew = np.ones((M,N))
        '''
        alpha = 0.01
        ep_max = 100
        epsilon = 0
        t_steps = 30
        episodes = 500
        '''
        p = np.zeros((M,N))
        dist = [epsilon, 1-epsilon]
        choice = [1,2]
        
        def softmax(s):
            p_theta = np.exp(theta[:,s])
            return p_theta/np.sum(p_theta), np.argmax(p_theta/np.sum(p_theta))
        
        count= 0
        rew_p_st = np.zeros(episodes)
        #iteration = np.zeros_like(rew_p_st)      
        for j in range(episodes):
            count += 1
            R = np.zeros((ep_max,1))
            grd = np.zeros((ep_max,M,N))
            ep_sum = np.zeros((M,N))
            theta_old = theta
            for i in range(ep_max): 
                s = self.env.reset()
                for t in range(t_steps):          
                    tmp1, tmp2  = softmax(s)
                    
                    #tmp3 = [tmp1[0], tmp1[1], tmp1[2], tmp1[3]]
                    tmp = random.choices(choice_a, tmp1)
                    a = tmp[0]
                    
                    obs, reward, done, info = self.env.step(a)
                    rew_p_st[j] = rew_p_st[j] + reward
                    R[i] += reward
                    phi = np.zeros((M,N))
                    phi[a,s] = rew[a,s]
                    grd[i,:,:] += phi 
                    p_theta, a = softmax(s)
                    
                    for ac in range(M):
                        phi = np.zeros((M,N))
                        phi[ac,s] = rew[ac,s]
                        grd[i,:,:] -= p_theta[ac]*phi
                    
                    s = obs
              
                ep_sum += (1/ep_max)*R[i]*grd[i,:,:] 
         
            #print('Count is: ', count)
            theta = theta_old + alpha*ep_sum
        
            #print(np.linalg.norm(theta-theta_old))
            rew_p_st[j] = rew_p_st[j]/(ep_max*t_steps)
        
        return rew_p_st