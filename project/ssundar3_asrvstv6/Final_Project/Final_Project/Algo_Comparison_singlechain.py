# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:04:28 2018

@author: baranwa2
"""
import numpy as np
import importlib, random
import matplotlib.pyplot as plt
import gym_env
import gym
from RL_Agents import sarsa, Risca, single_chain_REPS_class_version, Reinforce

sarsa = importlib.reload(sarsa)
Risca = importlib.reload(Risca)
single_chain_class_version = importlib.reload(single_chain_REPS_class_version)
Reinforce = importlib.reload(Reinforce)

#Creating object of Sarsa class and passing environment
env = gym.make('single_chain-v0')
sarsa_agent = sarsa.SarsaAgent(env)

#-----------------SARSA Parametrs-----------
epsilon = 1
alpha = 0.1
gamma = 0.9
episode_max_length = 1000
num_episodes = 10
#count_var = int(num_episodes/500)
#new_var = count_var+1
tot_reward_sarsa = np.zeros((5,num_episodes))
episode_50 = [i for i in range(num_episodes)]
#np.random.seed(0)

#------------REPS Parameters--------------------
#features = [[1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]]
features = [[1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9]]

features = np.array(features)

theta0 = np.dot(10,np.random.random_sample(features[0].shape[0]))
eta0 = 10
x0 = np.append(theta0, eta0)
epsilon_r = 1

#R = np.array([[0, 0], [0, 2], [0, 2], [0, 2], [10, 2]])
R = np.array([[0, 0], [0, 2], [0, 2], [10, 2]])
info_loss=0.5
n_policy_updates=10
n_samples=500
opt_method='L-BFGS-B'
tot_reward_reps = np.zeros((5,n_policy_updates))
episode_reps = [i for i in range(n_policy_updates)]

#----Policy Gradient Parameters----------
reinforce_agent = Reinforce.ReinforceAgent(env)
risca_agent = Risca.ReinforceCis(env)
alphan = 0.01
ep_max = 25
epsilonn = 0
t_steps = 40
episodes = 10
tot_reward_reinforce = np.zeros((5,episodes))
#episode_reps = [i for i in range(n_policy_updates)]
tot_reward_risca = np.zeros((5,episodes))

for item in range(5):
    rewards_sarsa = []
    rewards_reinforce = []
    rewards_risca = []
    rewards_reps = []
    policy = single_chain_class_version.Policy_REPS(env)
    agent = single_chain_class_version.run_REPS()
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
plt.title('Performance of various algorithms on Two State environment')
plt.xlabel('Policy Updates')
plt.ylabel('Average Reward')
label_line = ['SARSA','REPS','REINFORCE','REINFORCE_IS_CA']
plt.legend([line1,line2,line3,line4], label_line)
plt.show()
