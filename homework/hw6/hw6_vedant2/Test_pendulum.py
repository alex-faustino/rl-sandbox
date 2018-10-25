# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 04:15:13 2018

@author: Vedant
"""

import gym,sys
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

import Agents.PPO as PPO

class Neural_net(nn.Module):

    def __init__(self,in_layer = 3, out_layer = 1):
        super(Neural_net, self).__init__()
        self.fc1 = nn.Linear(in_layer,5)
        self.fc2 = nn.Linear(5,3)
        self.fc3 = nn.Linear(3,3)
        self.fc4 = nn.Linear(3,3)
        self.head = nn.Linear(3, out_layer)

    def forward(self, x):
        #x = x.view(-1, self.num_flat_features(x))
        #x = x.transpose(0,1)
        x = F.relu(self.fc1(x))
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc2(x))
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)
        
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
env = gym.make('Pendulum-v0')

policy = Neural_net(3,1)

algo = PPO.PPO(policy)
