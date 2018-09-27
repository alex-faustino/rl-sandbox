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
    def __init__(self, env, input_size = 4, output_size = 3, alpha = 0.9, GAMMA = 0.999, episode_max_length = 25, num_episodes = 1000, BATCH_SIZE = 128, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200, TARGET_UPDATE = 10):
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
    '''
    def reset(self):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    '''
    def select_action(self, state):
        #print('state = ', state)
        #print("IIIIII")
        action=0
        #print("IIIIII")

        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if np.random.uniform(0, 1) > self.epsilon:
            with torch.no_grad():
                #print(self.policy_net(state))
                action = self.policy_net(state).max(1)[1].view(1, 1)
                #print("in if")
        else:
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.long)
            #print("in else")
        #print("IIIIII")
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
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        #print(state_batch)
        print(self.policy_net(state_batch).gather(1, action_batch))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    def train(self):
        self.policy_net = DQN(self.input_size, self.output_size)
        self.target_net = DQN(self.input_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        optimizer = optim.Adam(self.policy_net.parameters())
        memory = ReplayMemory(10000)
        for i_episode in range(self.num_episodes):
            self.state = self.env.reset()
            #self.state = Variable(torch.from_numpy(self.state))
            self.state = torch.tensor([self.state]).float()

            #print("state = ",self.state)
            #print(self.policy_net(self.state))
            #print(self.policy_net(self.state).max(1)[0].view(1, 1))
            #print("max = ",self.policy_net(self.state).max(0)[0].view(1, 1))
            #action = self.policy_net(self.state).max(-1)[0].view(1, 1)
            #print(self.state)
            #print('episode = ', i_episode)
            #print(self.state)
            for t in range(self.episode_max_length):
                #print("t = ", t)
                action = self.select_action(self.state)
                next_state, reward, done, info = self.env.step(action)
                state = torch.tensor(self.state)
                action = torch.tensor([action])
                #print("next state = ", next_state)
                next_state = torch.tensor([next_state]).float()
                #print("next = ", next_state)
                reward = torch.tensor([reward])
                if done:
                    next_state = None
                memory.push(state, action, next_state, reward)
                self.state = next_state
                self.optimize_model(memory)
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()
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
