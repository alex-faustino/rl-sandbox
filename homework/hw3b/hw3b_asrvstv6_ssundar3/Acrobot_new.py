# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:30:13 2018

@author: baranwa2
"""

import gym
from gym import spaces
from gym.utils import seeding
import gym_envs
#import random
#from random import choices
import matplotlib.pyplot as plt
import numpy as np
from DQN_new import DeepQNetworkfinal

max_episodes = 20000
max_epochs = max_episodes/100
max_iterations = 500 

cum_reward = max_episodes*[0.]
avg_epreward = []
episode_count =  max_episodes*[0]
#avgepisode_reward = (max_epochs)*[0.]

if __name__ == "__main__":
    env = gym.make('acrobot-v0')
    RL = DeepQNetworkfinal(3, 6, e_greedy_increment=0.05, output_graph = True)
    
    for episode in range(max_episodes):
        # intial observation
        observation = env.reset()
        step = 0
        for iteration in range(max_iterations):
            #env.render()
            
            action = RL.choose_action(observation)
            #print(action)
            #observation_, reward, done,_  = env.step(action)
            observation_, reward,done = env.step(action)
            cum_reward[episode]+= reward
            #print(reward)
            #print(observation_)
            if iteration== max_iterations-1:
                done = True
                
            RL.store_transition(observation, action, reward, observation_,int(done))
            
            if (step > 50) and (step % 5 == 0):
                RL.learn()
                
            observation = observation_
            
            if done:
                break
            step += 1
        cum_reward[episode] = cum_reward[episode]/(iteration+1)
        print('The avg reward after episode %d is %f '%(episode,cum_reward[episode]))
        episode_count[episode] = episode
        if episode % 100 == 0 and episode > 0:
            avg_epreward.append(np.mean(cum_reward[-100:]))
            print('Episodes:',episode,'Average Reward:',avg_epreward[-1])
            
    plt.figure(1)
    plt.plot(episode_count,cum_reward)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.figure(2)
    plt.plot(avg_epreward)
    plt.xlabel('Episode')
    plt.ylabel('Reward averaged over 100 episodes')
    RL.plot_cost()
    plt.show()