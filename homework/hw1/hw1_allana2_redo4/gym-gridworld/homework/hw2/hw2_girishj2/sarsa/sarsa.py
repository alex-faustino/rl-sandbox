'''
# ------------------------------------------
AE 598RL Homework-2
Author: Girish Joshi
Email: girishj2@illinois.edu
This Code implements the SARSA Learning Algorithm
#-------------------------------------------
'''
import numpy as np
import random

# set the random seed
np.random.seed(1234)

class Sarsa:
    def __init__(self,s_dim,a_dim, lr = 0.05, gamma = 0.99):
        self.Q = -1000.*np.ones((s_dim,a_dim), dtype=float)
        self.lr = lr
        self.gamma = gamma
        self.a_dim = a_dim

    def learnQ(self, state, action, reward, next_state, next_action):
        oldQ = self.Q[state,action]
        value = self.Q[next_state, next_action]
        if oldQ == -1000:
            self.Q[state, action] = reward
        else:
            self.Q[state,action] = oldQ + self.lr *(reward + self.gamma*value - oldQ)

    def eGreedyAction(self, state, epsilon):
        if np.random.uniform(low=0.0, high=1.0, size=None) < epsilon:
            action = np.random.randint(0,self.a_dim)
        else:
            q = []
            for actions in range(self.a_dim):
                q.append(self.Q[state, actions])
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                bestQ = [i for i in range(self.a_dim) if q[i] == maxQ]
                i  = random.choice(bestQ)
            else:
                i = q.index(maxQ)
            action = i
        return action    