import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.head = nn.Linear(6, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class DQNAgent(object):
    def __init__(self, env, input_size = 4, output_size = 3, alpha = 0.9, GAMMA = 0.999, episode_max_length = 25, num_episodes = 50, BATCH_SIZE = 128, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200, TARGET_UPDATE = 10):
        self.env = env
        self.epsilon = EPS_START
        self.alpha = alpha
        self.GAMMA = GAMMA
        self.episode_max_length = episode_max_length
        self.num_episodes = num_episodes
        self.BATCH_SIZE = BATCH_SIZE
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.input_size = input_size
        self.output_size = output_size
        self.steps_done = 0
    def select_action(self, state):
        sample = random.random()
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.long)
        return action
    def optimize_model(self, memory):
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)#.float()
        action_batch = torch.cat(batch.action).view(self.BATCH_SIZE, 1)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def train(self):
        self.policy_net = DQN(self.input_size, self.output_size)
        self.target_net = DQN(self.input_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        memory = ReplayMemory(10000)
        reward_array = []
        for i_episode in range(self.num_episodes):
            print("i = ", i_episode)
            self.state = torch.from_numpy(self.env.reset()).float().view(1,4)
            reward_current = 0
            for t in range(self.episode_max_length):
                action = self.select_action(self.state)
                next_state, reward, done, info = self.env.step(action)
                reward_current += reward
                state = torch.tensor(self.state)
                action = torch.tensor([action])
                next_state = torch.tensor([next_state]).float().view(1,4)
                reward = torch.tensor([reward])
                if done:
                    next_state = None
                memory.push(state, action, next_state, reward)
                self.state = next_state
                self.optimize_model(memory)
            print("current reward = ", reward_current)
            reward_array.append(reward_current)
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        #self.env.render()
        #self.env.close()
        plt.plot(reward_array)
        plt.xlabel('Number of episodes')
        plt.ylabel('Episode Reward')
        plt.show()
        #plt.ioff()
        #plt.show()
        #return (rewards, states, actions)

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
