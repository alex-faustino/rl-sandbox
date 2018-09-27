#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:32:49 2018

@author: vedant
"""

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)   
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
for i in range(5):
    observation = np.array([np.random.randint(10),np.random.randint(10),np.random.randint(10)])
    action  = np.array([np.random.randint(10),np.random.randint(10),np.random.randint(10)])
    reward  = np.random.randint(10) 
    memory.push(observation, action, observation, reward)

BATCH_SIZE = 2
transitions = memory.sample(BATCH_SIZE)

batch = Transition(*zip(*transitions))

non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.uint8)


#non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
state_batch = torch.cat(batch.state)
action_batch = torch.cat(batch.action)
reward_batch = torch.cat(batch.reward)
