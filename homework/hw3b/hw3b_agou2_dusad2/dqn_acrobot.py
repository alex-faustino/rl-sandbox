#!/usr/bin/env python
# coding: utf-8

# In[8]:

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
from acrobot import AcrobotEnv
import numpy as np
from tqdm import tqdm, trange, tqdm_notebook
from collections import namedtuple
import random
import sys


# Heavily referencing this link
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# In[9]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x        

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# In[11]:


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.*steps_done/EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return net(state).argmax()
    else:
        return torch.tensor(random.randrange(3))


# In[12]:


def to_tensor(env_state):
    return torch.from_numpy(env_state).float()


# In[14]:


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
num_episodes = 50


# In[17]:


env = AcrobotEnv()

# Initialize replay memory
memory  = ReplayMemory(10000)

# Initialize action-value function Q
net = Net()

# initialize target action-value function Q'
clone = Net()
clone.load_state_dict(net.state_dict())
clone.eval()

optimizer = optim.RMSprop(net.parameters())

# net = torch.load("original.pth")
# clone = torch.load("target.pth")
# For each episode
def train(env, net, clone, memory, BATCH_SIZE = 64, GAMMA = 0.999, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200, num_episodes = 50,num_steps=1000, save_model=False):
    states_ = {}
    rewards_ = {}
    for i in tqdm_notebook(range(num_episodes), desc="Episode", file=sys.stdout):
        state = env.reset()
        rewards_[i] = []
        states_[i] = []
        states_[i].append(state)
        for j in tqdm_notebook(range(num_steps), desc="Step", file=sys.stdout):
            # Select action according to epsilon greedy 
            action = select_action(to_tensor(state))

            # execution action and observe reward and next_state
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor(reward)
            rewards_[i].append(reward)
            # store transition in memory
            memory.push(to_tensor(state), action, to_tensor(state), reward)

            state = next_state
            states_[i].append(state)
            if len(memory) >= BATCH_SIZE:
                # sample random minibatch of transitions
                minibatch = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*minibatch))
                state_batch = torch.stack(batch.state)
                action_batch = torch.stack(batch.action).view((64,1))
                reward_batch = torch.stack(batch.reward)

                # compute state values
                state_action_values = net(state_batch).gather(1, action_batch)

                # compute y_j
                next_state_values = clone(state_batch).max(1)[0]
                y_j = reward_batch.float() + GAMMA*next_state_values

                criterion = nn.MSELoss()
                loss = criterion(state_action_values, y_j)
                optimizer.zero_grad()
                loss.backward()
                # clip error term between -1 and 1 
                for param in net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            if done:
                break
        clone.load_state_dict(net.state_dict())
    if save_model:
        torch.save(clone, "target_{}_{}_{}.pth".format(BATCH_SIZE, num_episodes, num_steps))
        torch.save(net, "original_{}_{}_{}.pth".format(BATCH_SIZE, num_episodes, num_steps))
    return states_, rewards_


def test():
    # In[ ]:
    state = env.reset()
    rewards = []
    for i in tqdm(range(1000)):
        action = select_action(to_tensor(state))
        state, reward, done, _ = env.step(action.item())
        env.render()
        rewards.append(reward)

    
    

    plt.figure()
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.plot(rewards)
    plt.show()  



