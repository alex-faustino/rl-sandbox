# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 02:14:59 2018

@author: Vedant
"""
#from joblib import Parallel, delayed
import gym,sys
import attitudeDynamics
import numpy as np

from collections import namedtuple
import multiprocessing
num_cores = multiprocessing.cpu_count()
import matplotlib.pyplot as plt

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

gamma=0.99
seed=0
render=False
log_interval=20
torch.manual_seed(seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])
torch.set_default_tensor_type('torch.DoubleTensor')
inner_neuron = 50
class ActorCriticNet(nn.Module):


    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(7, inner_neuron)
        self.fc2 = nn.Linear(inner_neuron, inner_neuron)
        self.fc3 = nn.Linear(inner_neuron, inner_neuron)
        self.mu_head = nn.Linear(inner_neuron, 3)
        self.sigma_head = nn.Linear(inner_neuron, 3)
        self.v_head = nn.Linear(inner_neuron, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #if (inner_neuron>=10):
        #    x = F.dropout(self.fc1(x),0.2)
        x = F.relu(self.fc2(x))
        #if (inner_neuron>=10):
        #    x = F.dropout(self.fc2(x),0.2)
        x = F.relu(self.fc3(x))
        #if (inner_neuron>=10):
        #    x = F.dropout(self.fc3(x),0.2)
        mu = 2.0* (torch.tanh(self.mu_head(x)))
        sigma = F.softplus(self.sigma_head(x))
        state_value = self.v_head(x)
        return (mu, sigma,state_value)


class Agent():

    clip_param = 0.3
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 500, 50

    def __init__(self):
        self.training_step = 0
        #self.anet = ActorNet().float()
        #self.cnet = CriticNet().float()
        self.acnet = ActorCriticNet().float()
        self.buffer = []
        self.counter = 0

        #self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        #self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)
        self.optimizer_ac = optim.Adam(self.acnet.parameters(), lr=1e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma,_) = self.acnet(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action.clamp(-2.0, 2.0)
        return action, action_log_prob

    def get_value(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            _,_,state_value = self.acnet(state)
        return state_value.item()

    def save_param(self):
        torch.save(self.acnet.state_dict(), 'ppo_anet_params.pkl')
        #torch.save(self.cnet.state_dict(), 'ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            _,_,temp3 = self.acnet(s_)
            target_v = r + gamma * temp3

        _,_,temp = self.acnet(s) 
        adv = (target_v - temp).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma,_) = self.acnet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                _,_,temp2 = self.acnet(s[index])
                value_loss = F.smooth_l1_loss(temp2, target_v[index])
                ac_loss = action_loss+value_loss
                
                self.optimizer_ac.zero_grad()
                ac_loss.backward()
                nn.utils.clip_grad_norm_(self.acnet.parameters(), self.max_grad_norm)
                self.optimizer_ac.step()
                '''
                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()
                '''

        del self.buffer[:]


def main():
    #env = gym.make('Pendulum-v0')
    env = gym.make('attitudeDynamics-v0')
    env.seed(seed)
    end = 0
    render = 0
    agent = Agent()
    last_state = []
    training_records = []
    running_reward = -1000
    state = env.reset()
    for i_ep in range(1000):
        score = 0
        
        state = env.reset()
        nanerror = 1
        STA_q = []
        STA_w = []
        
        for t in range(5000):
            
            action, action_log_prob = agent.select_action(state)
            if np.isnan([action.numpy()[0][0]]):
                print('NAN!')
                nanerror = 1
                agent.save_param()
                break
            state_, reward, done, _ = env.step([action.numpy()[0][0]])
            if render:
                env.render()
            if agent.store(Transition(state, action[0].tolist(), action_log_prob[0].tolist(), (reward + 8) / 8, state_)):
                agent.update()
            score += reward
            state = state_
            STA_q.append(state[0:4])
            STA_w.append(state[4:7])
            

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % (log_interval/10) == 0:
            print('Ep {}\tMoving average score: {:.2f}, Current score : {:.2f}\t'.format(i_ep, running_reward,score))
        if i_ep % (0.1*log_interval) == 0:
            plt.plot(np.array(STA_q))
            plt.title('Quaternion')
            plt.xlabel('time_step')
            plt.ylabel('states')
            plt.show()
            #plt.savefig("img/ppo.png")
            plt.plot(np.array(STA_w))
            plt.title('Omega')
            plt.xlabel('time_step')
            plt.ylabel('states')
            #plt.savefig("img/ppo.png")
            plt.show()
                    #print("Solved! Moving average score is now {}!".format(running_reward))
            #render = 1
        if running_reward >-20:
            end = 1
        if end:    
            #env.close()
            break
            
    #save_param()
    if nanerror == 0:
        agent.save_param()
    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    #plt.savefig("img/ppo.png")
    plt.show()
    


if __name__ == '__main__':
    main()