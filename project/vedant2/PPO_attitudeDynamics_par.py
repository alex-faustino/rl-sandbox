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
import sys

gamma=0.99
seed=0
render=False
log_interval=4
torch.manual_seed(seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])
torch.set_default_tensor_type('torch.DoubleTensor')
inner_neuron = 100
class ActorCriticNet(nn.Module):


    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(7, inner_neuron)
        self.fc2 = nn.Linear(inner_neuron, inner_neuron)
        self.fc3 = nn.Linear(inner_neuron, inner_neuron)
        self.fc4 = nn.Linear(inner_neuron, inner_neuron)
        self.fc5 = nn.Linear(inner_neuron, inner_neuron)
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
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        mu = 2.0* (torch.tanh(self.mu_head(x)))
        sigma = F.softplus(self.sigma_head(x))
        state_value = self.v_head(x)
        return (mu, (sigma+1e-10),state_value)


class Agent():

    clip_param = 0.2
    max_grad_norm = 0.3
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 250

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
        #print('R.std : ',r.std())
        with torch.no_grad():
            _,_,temp3 = self.acnet(s_)
            target_v = r + gamma * temp3

        _,_,temp = self.acnet(s) 
        adv = (target_v - temp).detach()
        #print('Adv : ',adv)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma,_) = self.acnet(s[index])
                
                #sigma = torch.clamp(sigma, min=1e-5, max=10)
                dist = Normal(mu, sigma)
                #print('mu: ',mu, ' sigma: ',sigma)
                action_log_probs = dist.log_prob(a[index])
                #print('action_log_probs: ',action_log_probs)
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])
                #print('Ratio: ', ratio)
                if np.isnan(action_log_probs.detach().numpy()).any():
                    print('mu NAN!')
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                _,_,temp2 = self.acnet(s[index])
                value_loss = F.smooth_l1_loss(temp2, target_v[index])
                ac_loss = action_loss+value_loss
                #print('AC loss : ',ac_loss)
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

def rollout(params):
    env,end ,render ,agent ,last_state,training_records,running_reward,state,time_lim,i_ep,plot = params
    env = env[i_ep]
    if plot:
        STA_q = []
        STA_w = []
    score = 0
    
    for t in range(time_lim):
        action, action_log_prob = agent.select_action(state)
        state_, reward, done, _ = env.step([action.numpy()[0][0]])
        if render:
            env.render()
        if agent.store(Transition(state, action[0].tolist(), action_log_prob[0].tolist(), (reward), state_)):
            agent.update()
        score += reward
        state = state_
        if plot:
            STA_q.append(state[0:4])
            STA_w.append(state[4:7])
        

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

    if ((i_ep % (log_interval/(2)) == 0) & plot):
        print('Ep {}\tMoving average score: {:.2f}, Current score : {:.2f}\t'.format(i_ep, running_reward,score))
    if ((i_ep % (log_interval) == 0) & plot):
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
    return running_reward
    
    
def main():

    
    
    end = [0]*(N-1)
    render = [0]*(N-1)
    agent = [Agent()]*(N-1)
    last_state = []*(N-1)
    training_records = []*(N-1)
    running_reward = [-1000]*(N-1)
    Envlist = []*(N-1)
    state = []*(N-1)
    for i in range(len(Envlist)):
        Envlist[i] = gym.make('attitudeDynamics-v0')
        state[i] = Envlist[i_ep].reset()
    
    time_lim = [2500]*(N-1);
    j = range(N-1)
    for i_ep in range(round(5000/(N-1))):
        
        plot = [False]*(N-1)
        
        if (i_ep%log_interval == 0):
            plot[0] = True
        z = zip(Envlist,end ,render ,agent ,last_state,training_records,running_reward,state,time_lim,j,plot)
        F=p.map(rollout,z)
        print(F)
        #running_reward = rollout(Envlist[i_ep],end ,render ,agent ,last_state,training_records,running_reward,state,time_lim,i_ep,plot)
        F = np.array(F)
        if F.any() >-20:
            end = 1
        if end:    
            #env.close()
            break    
    #save_param()
    agent[0].save_param()
    '''
    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    #plt.savefig("img/ppo.png")
    plt.show()
    '''


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    N = multiprocessing.cpu_count()
    p = multiprocessing.Pool(N-1)
    main()