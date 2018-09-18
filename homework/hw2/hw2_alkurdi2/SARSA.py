# -*- coding: utf-8 -*-
"""
Abdul Alkurdi
AE598
SARSA
"""

from random import random
import numpy as np

class Sarsa(object):
    def __init__(self, env):
        self.env=env
        self.reset()
    
    def reset(self):
        self.Q=np.zeros((self.env.observation_space.n,self.env.action_space.n))
    def train(self, epsilon, alpha, gamma, epsilon_max, episode_length,epis_num):
        self.reset()
        rewards=np.zeros(epis.num)
        for i in range(epis.num):
            s=self.env.reset()
            a=self.action_epsilon_G(s,epsilon)
            for j in range(episode_length):
                (s_new,r,done)=self.env.step(a)
                rewards[i] += r
                a_new=self.action_epsilon_g(s_new,epsilon)
                self.Q[s,a] += alpha*(r+gamma*self.Q[s_new,a_new] - self.Q[s,a])
                (s,a)=(s_new,a_new)
                if done:
                    break
        return rewards
    def action_greedy(self, s):
        return self.Q[s,:].argmax()
    def action_random(self, s):
        return random.randrange(self.eng.action_space.n)
    def action_epsilon(self, s,epsilon):
        if random()<epsilon:
            return self.action_random(s)
        return self.action_greedy(s)
    def enjoy(self, episode_max_length):
        reward=np.zeros(episode_length)
        states=np.zeros(episode_length,dtype=int)
        actions=np.zeros(episode_length,dtype=int)
        s=self.env.reset()
        for j in range(episode_length):
            a=self.action_greedy(s)
            states[j]=s
            actions[j]=a
            (s_new,r,done)=self.env.step(a)
            rewards[j]=r;
            s=s_new
            if done:
                break
        return (rewards,states,actions)