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
        self.num_episodes = int(1000)#int(10000) #(using shorter length for faster testing)
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
        self.my_state_log = np.random.randint(0,7+1,size=(2,self.episode_length*self.num_episodes))
        self.my_action_log = np.random.randint(0,1, size=(1,self.episode_length*self.num_episodes))
        self.episode_counter = 0
        self.replay_length = 250#250 is edge of reasonable for run-time, let's see how this affects results
        self.minibatch_size = my_nn.reportMinibatchSize()#using same minibatch size for my_nn and my_nn2
        self.minibatch_log = np.random.rand(4,self.minibatch_size)#state,action,reward, next state
        self.minibatch_index = np.random.randint(0,1,size=self.minibatch_size)
        self.replay_index = np.random.randint(0,1,size=self.replay_length)
        pass
    def my_policy(self):
        if self.update_q_label > 1:
            self.previous_action = self.action # still want this because the agent selected a poor action and should evaluate it
        self.action = int(self.allowed_actions[0,np.argmax(self.my_q_function[self.location_x+self.gridnum*self.location_y+(self.allowed_actions-1)*self.gridnum**2])])
        self.my_exploit_action_log[self.location_y,self.location_x] = int(self.action)
        if np.random.rand() <= self.my_epsilon:
         self.action = self.allowed_actions[0,np.random.randint(1,self.allowed_actions.shape[1])]
        pass# self.location_y,self.location_x,self.action
    def work(self):
        if self.render_label == 'render':
            fig, (ax) = self.env.render_init(self.render_label)
        k=0 # counter for episodic cumulative reward
        for i in range(0,self.episode_length * self.num_episodes):
            self.my_reward_log[0,i] = self.my_reward[self.location_y,self.location_x]

## set location back "inbounds" before storing state ##
            if self.my_reward[self.location_y,self.location_x] < 0:
              self.location_y = self.previous_y
              self.location_x = self.previous_x

            self.my_state_log[:,i] = np.array([int(self.location_x),int(self.location_y)])[np.newaxis]
            self.my_action_log[0,i] = int(self.action)#inaccuracy in first time step of each episode is not important because algorithm does not update on first episode
            if self.update_q_label > 1:

## set experience replay and minibatch ##
             self.replay_index = np.random.randint(np.maximum(i-self.replay_length,0),i,size=self.replay_length)
             self.minibatch_index = np.random.randint(0,self.replay_length,size=self.minibatch_size)

### ensure that i-self.replay_index is not the first or last step of an episode ###
             while (np.any(np.mod(self.replay_index-1,self.episode_length) == 0)) + (np.any(np.mod(self.replay_index,self.episode_length) == 0)):
               self.replay_index[( (np.mod(self.replay_index-1,self.episode_length) == 0) + (np.mod(self.replay_index,self.episode_length) == 0) )]= np.random.randint(0,np.minimum(i+1,self.replay_length),size=np.sum( ( (np.mod(self.replay_index-1,self.episode_length) == 0) + (np.mod(self.replay_index,self.episode_length) == 0) ) ) )

### update minibatch ###
             self.minibatch_log[1,:] = self.my_action_log[0,self.replay_index[self.minibatch_index]]#stores action to get to state of same index
             self.minibatch_log[0,:] = self.my_state_log[0,self.replay_index[self.minibatch_index]-1]+self.gridnum*self.my_state_log[1,self.replay_index[self.minibatch_index]-1] #self.my_state_log[0,self.replay_index[self.minibatch_index]-1]+self.gridnum*self.my_state_log[1,self.replay_index[self.minibatch_index]-1]+self.gridnum**2*(self.minibatch_log[1,:]-1)
             self.minibatch_log[2,:] = self.my_reward_log[0,self.replay_index[self.minibatch_index]]
             self.minibatch_log[3,:] = self.my_state_log[0,self.replay_index[self.minibatch_index]]+self.gridnum*self.my_state_log[1,self.replay_index[self.minibatch_index]]#action selected in nn via np.amax, so not included

## train main neural network ##
             temporary_nn_main_input = self.minibatch_log[3,:][np.newaxis]
             self.my_nn.update(np.transpose(temporary_nn_main_input),self.minibatch_log[2,:],self.my_gamma,self.minibatch_log[0,:],self.minibatch_log[1,:].astype(int))

## train target neural network (logic in class file to only update when appropriate) ##
             self.my_nn2.update(self.my_nn.transmitModel(),i+1)

## allows learning (should be called after updates of Q function to ensure updating only under healthy circumstances) ##
            self.update_q_label += 1# update q function after first iteration

## output main neural network prediction of Q function using current location ##
            temporary_nn_main_output = self.my_nn.predict(np.transpose(self.location_x+self.gridnum*self.location_y)*np.ones(21))
            temporary_nn_main_output = temporary_nn_main_output.detach().numpy()
#            self.my_q_function[self.location_x+self.gridnum*self.location_y::self.gridnum**2] = temporary_nn_main_output# fixes sizing issue
            self.my_q_function[self.location_x+self.gridnum*self.location_y::self.gridnum**2] = temporary_nn_main_output[0]# fixes sizing issue

## implement agent policy (worked for tabular, no code changed) ##
            self.my_policy()
            (self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y) = self.env.step(self.my_reward, self.action, self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y)

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
