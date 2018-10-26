from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_plus1'])


class PPO:
    def __init__(self, env, gamma, clip, max_grad_norm, ppo_epoch, buffer_size, batch_size):
        self.env = env
        self.gamma = gamma
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.ppo_epoch = ppo_epoch
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.training_step = 0
        self.actor = Actor().float()
        self.critic = Critic().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=3e-4)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.critic(state)
        return state_value.item()

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_size == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_plus1 = torch.tensor([t.s_plus1 for t in self.buffer], dtype=torch.float)

        last_a_log_p = torch.tensor([t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + self.gamma * self.critic(s_plus1)

        adv = (target_v - self.critic(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_size)), self.batch_size, False):
                (mu, sigma) = self.actor(s[index])
                dist = Normal(mu, sigma)
                a_log_p = dist.log_prob(a[index])
                ratio = torch.exp(a_log_p - last_a_log_p[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.critic(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

        del self.buffer[:]

    def train(self, episodes_num, episode_length, viz):
        all_cum_rewards = []

        for episode in range(episodes_num):
            cum_reward = 0
            s = self.env.reset()

            for t in range(episode_length):
                if viz:
                    self.env.render()

                a, a_log_p = self.choose_action(s)
                s_plus1, r, done, _ = self.env.step([a])

                # update nets
                if self.store(Transition(s, a, a_log_p, (r + 8) / 8, s_plus1)):
                    self.update()

                # accumulate rewards
                cum_reward += r

                # cycle
                s = s_plus1

            all_cum_rewards.append(cum_reward)

        return all_cum_rewards


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

        return mu, sigma


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v(x)

        return state_value
