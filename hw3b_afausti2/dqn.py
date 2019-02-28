import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from DQN Pytorch tutorial
# See: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
# and https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Removed 3 convolution layers
# Eps annealing adjusted to match linear annealing in Mnih

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = "cpu"


class DQN(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size, reward_decay, eps_start, eps_end):
        super(DQN, self).__init__()
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.steps_done = 0

        self.fc1 = nn.Linear(in_channels, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def select_action(self, state):
        state = torch.tensor(state, device=device)
        sample = random.random()
        eps_threshold = (self.eps_start - self.eps_end)/(-10000.)*self.steps_done + self.eps_start
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.max(self(state), -1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]], device=device)

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
