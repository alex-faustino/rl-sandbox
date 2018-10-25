import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample(self, state):
        mean, var, value = self.forward(state)
        dist =  normal.Normal(mean, torch.abs(var)**0.5)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def eval(self, state, action):
        mean, var, value = self.forward(state)
        dist =  normal.Normal(mean, torch.abs(var)**0.5)
        log_prob = dist.log_prob(action)
        return log_prob, value

