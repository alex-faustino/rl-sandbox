# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 01:15:37 2018

@author: baranwa2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 23:51:23 2018

@author: baranwa2
"""

import numpy as np
import importlib, random
import matplotlib.pyplot as plt
import gym_env
import gym
from RL_Agents import sarsa, Risca, GridWorld_REPS_class_version, Reinforce

sarsa = importlib.reload(sarsa)
Risca = importlib.reload(Risca)
GridWorld_REPS_class_version = importlib.reload(GridWorld_REPS_class_version)
Reinforce = importlib.reload(Reinforce)
num_run = 5

#Creating object of Sarsa class and passing environment
env = gym.make('grid_world-v0')
sarsa_agent = sarsa.SarsaAgent(env)

#-----------------SARSA Parametrs-----------
epsilon = 1
alpha = 0.1
gamma = 0.9
episode_max_length = 1000
num_episodes = 50
#count_var = int(num_episodes/500)
#new_var = count_var+1
tot_reward_sarsa = np.zeros((num_run,num_episodes))
episode_50 = [i for i in range(num_episodes)]
#np.random.seed(0)

#------------REPS Parameters--------------------
#policy = two_state_class_version.Policy_REPS(env)
N = env.observation_space.n
M = env.action_space.n
features = [[i ** j for j in range(3)] for i in range(N)]
features = np.array(features)
theta0 = np.dot(10,np.random.random_sample(features[0].shape[0]))  # theta has the same shape of one feature vector.
eta0 = 10
epsilon_r = 1
x0 = np.append(theta0, eta0)
R = np.zeros((N,M))
R[0 : 5, 1] = -1
R[-5: ,0] = -1
R[0:21:5, 3] = -1
R[4:25:5,2] = -1
R[21,:] = 10
R[23,:] = 5
R = np.array(R)
info_loss=0.5
n_policy_updates= 50
n_samples= 100
opt_method='L-BFGS-B'
tot_reward_reps = np.zeros((num_run,n_policy_updates))
episode_reps = [i for i in range(n_policy_updates)]

#----Policy Gradient Parameters----------
reinforce_agent = Reinforce.ReinforceAgent(env)
risca_agent = Risca.ReinforceCis(env)
alphan = 0.01
ep_max = 25
epsilonn = 0
t_steps = 40
episodes = 50
tot_reward_reinforce = np.zeros((num_run,episodes))
#episode_reps = [i for i in range(n_policy_updates)]
tot_reward_risca = np.zeros((num_run,episodes))

for item in range(num_run):
    print('iteration no is %d'%item)
    rewards_sarsa = []
    rewards_reinforce = []
    rewards_risca = []
    rewards_reps = []
    policy = GridWorld_REPS_class_version.Policy_REPS(env)
    agent = GridWorld_REPS_class_version.run_REPS()
    rewards_sarsa = sarsa_agent.train(epsilon, alpha, gamma, episode_max_length, num_episodes)
    rewards_reps, V, policy, q_sa, theta, eta = agent.REPS(env, R, policy, features, x0, epsilon_r, info_loss,
                                      n_policy_updates, n_samples, opt_method)
    rewards_sarsa/= episode_max_length
    rewards_risca = risca_agent.train(alphan,ep_max,epsilonn,t_steps,episodes)
    rewards_reinforce = reinforce_agent.train(alphan,ep_max,epsilonn,t_steps,episodes)
    tot_reward_sarsa[item,:] = rewards_sarsa
    tot_reward_reps[item,:] = rewards_reps
    tot_reward_reinforce[item,:] = rewards_reinforce
    tot_reward_risca[item,:] = rewards_risca
    

mean1 = np.mean(tot_reward_sarsa, axis=0)
mean2 = np.mean(tot_reward_reps, axis=0)
mean3 = np.mean(tot_reward_reinforce, axis=0)
mean4 = np.mean(tot_reward_risca, axis=0)
#mean1[0] = np.amin(tot_reward[:,0])
green_point = dict(markerfacecolor='g', marker='D')
print('---SARSA_Means----')
print(mean1)
print(len(episode_50))
print('---REPS_Means---')
print(mean2)
print(len(episode_50))
print('---REINFORCE_Means---')
print(mean3)
print(len(episode_50))
print('---REINFORCE+IS+Causality_Means---')
print(mean4)
print(len(episode_50))

#Set_up for the plots
plt.figure(1)
plt.boxplot(tot_reward_sarsa,positions = episode_50 , showmeans = True)
line1, = plt.plot(episode_50,mean1,'b')
plt.boxplot(tot_reward_reps,positions = episode_50 , showmeans = True)
line2, = plt.plot(episode_50,mean2,'r')
plt.boxplot(tot_reward_reinforce,positions = episode_50 , showmeans = True)
line3, = plt.plot(episode_50,mean3,'c')
plt.boxplot(tot_reward_risca,positions = episode_50 , showmeans = True)
line4, = plt.plot(episode_50,mean4,'k')
plt.title('Performance of various algorithms on Grid World environment')
plt.xlabel('Policy Updates')
plt.ylabel('Average Reward')
label_line = ['SARSA','REPS','REINFORCE','REINFORCE_IS_CA']
plt.legend([line1,line2,line3,line4], label_line)
plt.show()