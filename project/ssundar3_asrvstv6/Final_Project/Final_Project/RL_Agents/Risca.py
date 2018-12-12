# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:26:48 2018

@author: sreen
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import gym_env
import random
from random import choices
import matplotlib.pyplot as plt


class ReinforceCis(object):
    
    def __init__(self,env):
        self.env = env
    
    def train(self,alpha,ep_max,epsilon,t_steps,episodes):
        M = self.env.action_space.n
        N = self.env.observation_space.n
        
        theta = np.zeros((M,N))
        rew = np.ones((M,N))
        '''
        alpha = 0.01
        ep_max = 100
        epsilon = 0
        t_steps = 30
        '''
        choice_a = [i for i in range(M)]
        
        def softmax(s,policy_g):
            
            #p_theta = np.exp(np.multiply(theta[:,s],rew[:,s]))
            p_theta = np.exp(policy_g[:,s])
            #print(np.linalg.norm(theta))
            return p_theta/np.sum(p_theta), np.argmax(p_theta/np.sum(p_theta))
        
        count= 0
        rew_p_st = np.zeros(episodes)
        #iteration = np.zeros_like(rew_p_st)    
        for j in range(episodes):
            
            tt = theta
            p_arr_old = np.zeros(t_steps)
            p_arr_new = np.zeros(t_steps)
            p_ratio = np.zeros(t_steps) 
            policy_new = theta
            for i in range(ep_max):
                
                grd = np.zeros((t_steps,M,N))
                s_arr = np.zeros(t_steps+1, dtype=np.int8) # to store the sequence of states in T steps
                a_arr = np.zeros(t_steps+1, dtype=np.int8) # to store the sequence of actions in T steps
                r_arr = np.zeros(t_steps+1) # to store the sequence of rewards in T steps.
                s_arr[0] = self.env.reset()
        
                for t in range(t_steps):
        #                
                    tmp1, tmp2  = softmax(s_arr[t],theta)
                    tmp = random.choices(choice_a, tmp1)
                    a_arr[t] = int(tmp[0])
                    p_arr_old[t] = tmp1[a_arr[t]]
                    tmp4,tmp5 = softmax(s_arr[t],policy_new)
                    p_arr_new[t] = tmp4[a_arr[t]]
                    p_ratio[t] = p_arr_new[t]/p_arr_old[t]
                    
                    obs, reward, done, info = self.env.step(a_arr[t])
                    rew_p_st[j] = rew_p_st[j] + reward
                    s_arr[t+1] = obs
                    r_arr[t+1] = reward
                    phi = np.zeros((M,N))
                   
                    phi[a_arr[t],s_arr[t]] = 1
                    grd[t,:,:] += phi 
                    p_theta, a_arr[t] = softmax(s_arr[t],theta)
                    
                    for ac in range(M):
                        phi = np.zeros((M,N))
                        phi[ac,s_arr[t]] = 1
                        grd[t,:,:] -= p_theta[ac]*phi
                    
                theta = policy_new
                sum1 = np.zeros_like(theta)
                for t in range(t_steps):
                    v_t = 0
                    for t1 in range(t+1, t_steps+1):
                        v_t += r_arr[t1]
                    delta = v_t
                    sum1 += alpha*delta*(grd[t,:,:])
          
                policy_new += np.prod(p_ratio)*sum1
        
            
            rew_p_st[j] = rew_p_st[j]/(ep_max*t_steps)
            #iteration[j] = j
        return rew_p_st
