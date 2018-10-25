## Development left to be done:
### Make replay memory and send over random sample to network

import numpy as np
import matplotlib.pyplot as plt
import pylab
import gym
import sys
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.ticker as plticker
np.set_printoptions(threshold=np.inf)

class qLearning(object):
    def __init__(self, env, my_nn, my_nn2, render_input): 
        self.env = env
        self.my_nn = my_nn
        self.my_nn2 = my_nn2
        self.render_label = render_input
        self.temporary_number, self.allowed_actions = env.states_and_actions()
        self.gridnum = int(np.sqrt(self.temporary_number))
        self.location_x = np.random.randint(int(1),self.gridnum-1)
        self.location_y = np.random.randint(int(1),self.gridnum-1)
        self.action = int(0) # no initial action until computed
        self.previous_action = self.action
        self.previous_x = self.location_x
        self.previous_y = self.location_y
        self.previous_previous_x = self.previous_x
        self.previous_previous_y = self.previous_y
        self.episode_length = int(100)
        self.num_episodes = int(10000) # currently only rumy_nning one episode, 10000
        self.my_alpha = 0.1
        self.my_gamma = 0.9
        self.my_epsilon = 0.1
        self.my_reward = np.array([ [-1,-1,-1,-1,-1,-1,-1],[-1, 0, 10, 0, 5, 0, -1],[-1, 0, 0, 0, 0, 0, -1],\
    [-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, -1, -1, -1, -1, -1, -1] ]) 
        self.my_reward = np.flipud(self.my_reward)
        self.my_reward_model = np.zeros([self.gridnum,self.gridnum])#reward model updated based on observations
        self.my_q_function = np.random.rand(self.gridnum**2*self.allowed_actions.shape[1])# randomly initialized via reward model
        self.my_q_function[0::7] = 0# to prevent biasing agent with favorable "out of bound" q functions
        self.my_square_q_function = np.random.rand(self.gridnum,self.gridnum,self.allowed_actions.shape[1])
        self.my_reward_log = np.random.rand(1, self.episode_length*self.num_episodes) # used to store reward for each time step
        self.my_episodic_cumulative_reward_log = np.random.rand(1,self.num_episodes)
        self.update_q_label = 0
        self.color_array = ['blue','orange']
        self.my_exploit_action_log = np.random.rand(self.gridnum,self.gridnum)
        self.my_state_log = np.random.rand(2,self.episode_length*self.num_episodes)
        self.my_action_log = np.random.rand(1, self.episode_length*self.num_episodes)
        self.episode_counter = 0
        self.replay_length = 60
        pass
    def my_policy(self):
        if self.update_q_label > 1:
            self.previous_action = self.action # still want this because the agent selected a poor action and should evaluate it
        self.action = self.allowed_actions[0,np.argmax(self.my_q_function[self.location_x+self.gridnum*self.location_y+(self.allowed_actions-1)*self.gridnum**2])]
        self.my_exploit_action_log[self.location_y,self.location_x] = int(self.action)
        if np.random.rand() <= self.my_epsilon:
         self.action = self.allowed_actions[0,np.random.randint(0,self.allowed_actions.shape[1])]
        pass# self.location_y,self.location_x,self.action
    def update_reward_model(self):# reversed y and x for reward (not model) to accommodate human-readable reward
        self.my_reward_model[self.location_y,self.location_x] = self.my_reward[self.location_y,self.location_x]
        pass
    def update_my_q_function(self):#,action,location_x,location_y):
        if self.update_q_label > 1:# update_q_label ensures that we do not update the q function when location is reset
            self.my_q_function[self.previous_x+self.gridnum*self.previous_y+(self.action-1)*self.gridnum**2] +=\
            self.my_alpha*(self.my_reward_model[self.previous_y,self.previous_x]+\
            self.my_gamma*np.amax(self.my_q_function[self.location_x+self.location_y*self.gridnum::self.gridnum**2])-\
            self.my_q_function[self.previous_x+self.gridnum*self.previous_y+(self.action-1)*self.gridnum**2])
        pass
    def work(self):
        if self.render_label == 'render':
            fig, (ax) = self.env.render_init(self.render_label)
        k=0 # counter for episodic cumulative reward
        for i in range(0,self.episode_length * self.num_episodes - 1):
#            self.update_reward_model()#used in tabular algorithm
            self.my_reward_log[0,i] = self.my_reward[self.location_y,self.location_x]
            self.my_state_log[:,i] = np.array([self.location_x,self.location_y])[np.newaxis]
            self.my_action_log[0,i] = self.action#inaccuracy in first time step of each episode is not important because algorithm does not update on first episode
#            if self.update_q_label > self.replay_length or (self.episode_counter>1 and self.update_q_label > 1):
            if self.update_q_label > 1: # I have reverted back to this boolean switch because I added a condition to the while loop. Should check this though
## first, draw sample from experience replay ##
             replay_index = np.random.randint(0,self.replay_length)
### ensure that i-replay_index is not the first or last step of an episode ###
             while (np.mod(i-replay_index-1+1,self.episode_length) == 0) or (np.mod(i-replay_index+1,self.episode_length) == 0) or replay_index > i-1:
              replay_index = np.random.randint(0,self.replay_length)

## "predicting" to store state in main network (reluNetworkClass.py) for use in target network (reluNetworkClass2.py) ##
             self.my_nn.predict(int(self.my_state_log[0,i-replay_index]+self.gridnum*self.my_state_log[1,i-replay_index]))

## train main neural network ## 
             self.my_nn.update(self.my_reward_log[0,i-replay_index-1], self.my_q_function[int(self.my_state_log[0,i-replay_index-1]+self.my_state_log[1,i-replay_index-1]*self.gridnum+self.gridnum**2*(self.my_action_log[0,i-replay_index-1]-1))], np.amax(self.my_q_function[int(self.my_state_log[0,i-replay_index]+self.my_state_log[1,i-replay_index]*self.gridnum)::self.gridnum**2]),self.my_gamma)

## train target neural network (logic in class file to only update when appropriate) ##
             self.my_nn2.update(self.my_nn.transmitModel(),i+1)#np.mod(i+1,self.episode_length)

## allows learning (should be called after updates of Q function to ensure updating only under healthy circumstances) ##
            self.update_q_label += 1# default is to update the q function after the first iteration

## set location back "inbounds" before taking predicting Q function and taking next step ##
            if self.my_reward[self.location_y,self.location_x] < 0:
              self.location_y = self.previous_y
              self.location_x = self.previous_x

## output main neural network prediction of Q function using current location ##
            self.my_q_function[self.location_x+self.gridnum*self.location_y::self.gridnum**2] = self.my_nn.predict(self.location_x+self.gridnum*self.location_y)

## implement agent policy ##
            self.my_policy() # closest to pragmatic results here
            (location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y) = self.env.step(self.my_reward, self.action, self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y)# current state is AFTER action
            (self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y) = (location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y)

## for new episode ##
            if np.mod(i+1,self.episode_length) == 0:
                self.my_episodic_cumulative_reward_log[0,k] = \
                np.sum(self.my_reward_log[0,(k*self.episode_length):(i+1)])# sums from k*episode_length to i
                k += 1
                self.env.reset()

## for real-time rendering ##
            if self.render_label == 'render':
                self.env.render(fig,ax,i,self.render_label)

## progress output ##
            progress_interval = 0.01# should be a percentage in a decimal form
            progress_checker = np.floor(progress_interval*self.episode_length*self.num_episodes)
            self.episode_counter = np.floor((i+1)/self.episode_length)
            if np.mod(i+1,progress_checker) == 0:
                sys.stdout.write("\r"+"%s" % int(progress_interval*100+np.floor(i/progress_checker)*progress_interval*100) + '%')#updates progress %
        sys.stdout.write("\r"+'done' + '\n')#displays progress and prints results on new lines
## results ##
        fig1, (ax1)=plt.subplots()
        ax1.plot(self.my_episodic_cumulative_reward_log[0,0:-1])
        plt.xlabel('episode number')
        plt.ylabel('total episodic reward')
#        print('reward model' + '\n' + str(np.flipud(self.my_reward_model[1:6,1:6])))#only relevant for tabular algorithm
        for a_index in range(0,self.allowed_actions.shape[1]):
         for y_index in range(0,self.gridnum-1):
          for x_index in range(0,self.gridnum-1):
           self.my_square_q_function[x_index,y_index,a_index] = self.my_q_function[x_index+y_index*self.gridnum+(a_index)*self.gridnum**2]
        print('Q function for up action' + '\n' + str(np.flipud(np.transpose(self.my_square_q_function[1:6,1:6,0]))))
        print('Q function for down action' + '\n' + str(np.flipud(np.transpose(self.my_square_q_function[1:6,1:6,1]))))
        print('Q function for left action' + '\n' + str(np.flipud(np.transpose(self.my_square_q_function[1:6,1:6,2]))))
        print('Q function for right action' + '\n' + str(np.flipud(np.transpose(self.my_square_q_function[1:6,1:6,3]))))
        print('total reward = ' + str(np.sum(self.my_reward_log[0,-self.episode_length:-1])))
        fig2, (ax2) = plt.subplots()
        ax2.plot(self.my_reward_log[0,-self.episode_length:-1])
        plt.xlabel('time step of episode')
        plt.ylabel('reward')
        fig3, (ax3) = plt.subplots()
        ax3.plot(np.transpose(self.my_state_log[0,-self.episode_length:-1]), label='x coordinate')
        ax3.plot(np.transpose(self.my_state_log[1,-self.episode_length:-1]), label='y coordinate')
        plt.xlabel('time step of episode')
        plt.ylabel('x and y coordinates')
        pylab.legend(loc='upper left')
        print('Exploit policy of agent, where: 1 is up, 2 is down, 3 is left and 4 is right')
        print(np.flipud(self.my_exploit_action_log[1:6,1:6]).astype(int))
        pass 
