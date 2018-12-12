# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:33:04 2018

@author: Abdul Alkurdi
SARSA algorithm adjusted for the final project
"""
import numpy as np
import gym
from random import random


def SARSAsimagent(meta, env, hard=False): 
    [epsilon,alpha,gamma,ep_len,ep_num]= meta    
    s_size=0
    a_size=0
    if type(env.action_space)==gym.spaces.discrete.Discrete:
        a_size=env.action_space.n
    else:
        a_size=np.prod(env.action_space.nvec)
    if type(env.observation_space)==gym.spaces.discrete.Discrete:
        s_size=env.observation_space.n
    else:
        s_size=np.prod(env.observation_space.nvec)
    reward=np.zeros(ep_num)  #sum of rewards at end of each episode
    #Q=np.random.rand(a_size,s_size)
    Q=np.zeros([a_size,s_size])

    big_state = []
    for i in range(ep_num):
        a_store=[]
        s_store=[]
        
        r_s=None
        r_s=[]
        s=env.reset()
        s_t=s[0]+s[1]*3
        if random() > epsilon:
                a=np.argmax(Q[:,s_t])
        else:
                a=env.action_space.sample()
        for j in range(ep_len):
            big_state.append(env.state)
            s_t=s[0]+5*s[1] #flattened state
            s_store.append(s_t)
            a_store.append(a)
            ns, r =env.step(a)#,hard)
            
            big_state.append(env.state)

            reward[i] += r
            r_s.append(r)
            ns_t=ns[0]+5*ns[1] 
            if random() > epsilon:
                na=np.argmax(Q[:,ns_t])
            else:
                na=env.action_space.sample()
            Q[a,s_t] += alpha*(r+ gamma* Q[na,ns_t]-Q[a,s_t])
            s=ns
            a=na
    return Q, reward, a_store, s_store, r_s, big_state

