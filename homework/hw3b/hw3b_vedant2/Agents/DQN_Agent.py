# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:34:57 2018

@author: Vedant
"""

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

torch.set_default_tensor_type('torch.DoubleTensor')

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
    
class DQN(nn.Module):

    def __init__(self,inp = 7):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inp,5)
        self.fc2 = nn.Linear(5,3)
        self.fc3 = nn.Linear(3,3)
        self.fc4 = nn.Linear(3,3)
        self.head = nn.Linear(3, 1)

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
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def DQN_Agent(env,input_layer = 7,BATCH_SIZE = 100,GAMMA = 0.99,TARGET_UPDATE = 20,initial_epsilon = 1, 
              final_epsilon = 0.01,total_episodes = 100, annealing_period = None,max_steps = 25
              , decay_rate = None,plot = False):
    
    if (annealing_period == None):
        annealing_period = total_episodes;
    if(annealing_period > total_episodes):
        print('Annealing period cannot be more than training period, setting annealing period equal to training period')
        annealing_period = total_episodes;
    
    epsilon = initial_epsilon;
    
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Q_net = DQN(input_layer).to(device)
    Q_target_net = DQN(input_layer).to(device)
    Q_target_net.load_state_dict(Q_net.state_dict())
    Q_target_net.eval()
    
    optimizer = optim.RMSprop(Q_net.parameters())
    memory = ReplayMemory(10000)
    
    def select_action(state,epsilon):
        action=np.array([-1.0,1.0])
        state_action_options = []
        for a in action:
            state_action = np.append(state,a)
            state_action_options.append(state_action)
        state_action_options = np.array(state_action_options)
        #state = state.reshape(state.shape+(1,))
        state = (torch.from_numpy(state))
        if np.random.uniform(0, 1) > epsilon:
            return action[torch.argmax(Q_net(torch.from_numpy(state_action_options)))]
            
        else:
            #bug
            return action[np.random.randint(2)]

    '''
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return Q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
    '''
    
    
    episode_durations = []
    
    
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
    
        plt.pause(0.001)  # pause a bit so that plots are updated
            
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.uint8)
        #nfns = [s for s in batch.next_state if s is not None]
        #non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None])
        state_batch = torch.tensor(batch.state)
        #action_batch = torch.tensor(batch.action)
        action_batch = torch.tensor([batch.action]).transpose(0,1)
        reward_batch = torch.tensor(batch.reward)
        sa_batch = torch.cat((state_batch,action_batch),1)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = Q_net(sa_batch)
    
        # Compute V(s_{t+1}) for all next states.
        #next_state_values = torch.zeros(BATCH_SIZE)
        #next_state_values[non_final_mask] = Q_target_net(non_final_next_states)
        next_state_values = Q_target_net(sa_batch)
        nsv_t = next_state_values.transpose(0,1)
        # Compute the expected Q values
        expected_state_action_values = (nsv_t * GAMMA) + reward_batch
        esav_nograd =  Variable(expected_state_action_values, requires_grad=False)
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, esav_nograd.transpose(0,1))
    
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in Q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
    av_r = []
    for i_episode in range(total_episodes):
        # Initialize the environment and state
        observation = env.reset()
        #last_screen = get_screen()
        #current_screen = get_screen()
        #state = current_screen - last_screen
        epi_reward = []
        #nstep = 100
        for t in range(max_steps):
            # Select and perform an action
            #
            if decay_rate != None:
                epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * np.exp(-decay_rate * t) 
            epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * (t+1)/max_steps
            action = select_action(observation,epsilon)
            new_observation, reward, done, info = env.step(action)
            #reward = torch.tensor([reward], device=device)
            epi_reward.append(reward)
            env.render()
            # Observe new state
            #last_screen = current_screen
            #current_screen = get_screen()
            '''
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            '''
            # Store the transition in memory
            memory.push(observation, action, new_observation, reward)
    
            # Move to the next state
            #state = next_state
            observation = new_observation
            # Perform one step of the optimization (on the target network)
            optimize_model()
            #if done:
        if plot:
            episode_durations.append(t + 1)
            plot_durations()
            #    break
        # Update the target network
        #plt.plot(np.array(epi_reward))
        if i_episode % TARGET_UPDATE == 0:
            Q_target_net.load_state_dict(Q_net.state_dict())
            
        av_r.append(np.array(epi_reward))
    print('Complete')
    return av_r
    #env.render()
    env.close()
    #plt.ioff()
    #plt.show()


        