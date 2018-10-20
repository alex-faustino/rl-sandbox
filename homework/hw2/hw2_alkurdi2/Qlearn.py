# -*- coding: utf-8 -*-
"""
Abdul Alkurdi
AE598
Qlearning Algorithm
"""
from random import random
import numpy as np
import gym
def Qlagent(meta, env, hard=False): 
    [epsilon,alpha,gamma,ep_len,ep_num]= meta    
    s_size=0
    a_size=0
    if type(env.action_space)==gym.spaces.discrete.Discrete:
        a_size=env.action_space.n
    elif type(env.action_space)==gym.spaces.multi_discrete.MultiDiscrete:
        a_size=np.prod(env.action_space.nvec)
    if type(env.observation_space)==gym.spaces.discrete.Discrete:
        s_size=env.observation_space.n
    elif type(env.observation_space)==gym.spaces.multi_discrete.MultiDiscrete:
        s_size=np.prod(env.observation_space.nvec)
    Q=np.zeros([a_size,s_size])
    reward=np.zeros(ep_num)
    for i in range(ep_num):
        a_store=[]
        s_store=[]
        s=env.reset()
        s_t=s[0]+s[1]*5
        for j in range(ep_len):
            s_t=s[0]+5*s[1] #one hot encoded state
            s_store.append(s_t)
            if random() > epsilon:
                a=np.argmax(Q[:,s_t])
            else:
                a=env.action_space.sample()
            a_store.append(a)
            ns, r, _ , _ =env.step(a,hard)
            ns_t=ns[0]+5*ns[1]
            reward[i] += r
            Q[a,s_t] += alpha*(r+ gamma* Q[np.argmax(Q[:,ns_t]),ns_t]-Q[a,s_t])
            s=ns
    return Q,  reward, a_store, s_store