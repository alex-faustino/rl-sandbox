import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import matplotlib.pyplot as plt
from collections import namedtuple

torch.manual_seed(0)

class Params():
    def __init__(self):
        self.batch_size = 64
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.max_grad = 0.5
        self.ppo_epoch = 10
        self.mem_size = 1000
        self.batch_size = 32
        self.num_steps = 2048
        self.num_episodes = 2000
        self.episode_max_length = 100
        self.seed = 1
        self.env_name = 'Pendulum-v0'

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.mu = nn.Linear(128, 1)
        self.sigma = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return (mu, sigma)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        valuefn = self.value(x)
        return valuefn

class ppo():
    def __init__(self, env):
        self.env = env
        self.training_step = 0
        self.actor = Actor().float()
        self.critic = Critic().float()
        self.buffer = []
        self.counter = 0
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        logprob = dist.log_prob(action)
        action.clamp(-2.0, 2.0)
        return action.item(), logprob.item()

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.mem_size == 0

    def update(self):
        self.training_step += 1
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        logprob = torch.tensor([t.logprob for t in self.buffer], dtype=torch.float).view(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        with torch.no_grad():
            valuefn = rewards + self.gamma * self.critic(next_state)
        advantage = (valuefn - self.critic(state)).detach()

        for pp in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.mem_size)), self.batch_size, False):
                (mu, sigma) = self.actor(state[index])
                dist = Normal(mu, sigma)
                new_logprob = dist.log_prob(action[index])
                pratio = torch.exp(new_logprob - logprob[index])

                L1 = pratio * advantage[index]
                L2 = torch.clamp(pratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[index]
                Lclip = -torch.mean(torch.min(L1, L2))

                self.actor_opt.zero_grad()
                Lclip.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                self.actor_opt.step()

                Lvf = F.smooth_l1_loss(self.critic(state[index]), valuefn[index])
                self.critic_opt.zero_grad()
                Lvf.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.critic_opt.step()
        del self.buffer[:]

    def train(self):
        param = Params()
        self.clip_param = param.clip
        self.max_grad = param.max_grad
        self.ppo_epoch = param.ppo_epoch
        self.mem_size = param.mem_size
        self.batch_size = param.batch_size
        self.gamma = param.gamma
        self.num_episodes = param.num_episodes
        self.episode_max_length = param.episode_max_length

        MemoryBuffer = namedtuple('MemoryBuffer', ['state', 'action', 'logprob', 'reward', 'next_state'])
        rewards = []
        for i_ep in range(self.num_episodes):
            reward_eps = 0
            state = self.env.reset()
            for t in range(self.episode_max_length):
                action, logprob = self.select_action(state)
                next_state, reward, done, info = self.env.step([action])
                if self.store(MemoryBuffer(state, action, logprob, (reward + 8) / 8, next_state)):
                    self.update()
                reward_eps += reward
                state = next_state
            rewards.append(reward_eps)
        plt.plot(rewards)
        plt.title('PPO')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
        #return store_avg_rewards
