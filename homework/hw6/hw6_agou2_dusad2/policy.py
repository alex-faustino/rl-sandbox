import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.shared_network = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128),  
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU())

        self.actor =  nn.Sequential(
            nn.Linear(64, 2))

        self.critic = nn.Sequential(
            nn.Linear(64, 1))

    def forward(self, x):
        temp = self.shared_network.forward(x)
        mu, var = self.actor.forward(temp)
        value = self.critic.forward(temp)
        sigma = torch.abs(var)**0.5
        return mu, sigma, value

    def sample(self, state):
        mean, sigma, value = self.forward(state)
        dist =  normal.Normal(mean, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def eval(self, state, action):
        mean, sigma, value = self.forward(state)
        dist =  normal.Normal(mean, sigma)
        log_prob = dist.log_prob(action)
        return log_prob, value

