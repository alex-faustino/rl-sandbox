'''
# ------------------------------------------
AE 598RL Homework-2
Author: Girish Joshi
Email: girishj2@illinois.edu
This Code implements the SARSA Algorithm on the Grid World Problem

The Grid world Environment is developed as part of Homework-1
Use the File: myGridworld.py
#-------------------------------------------
'''

from sarsa import Sarsa
from myGridworld import gridWorldEnv
import matplotlib.pyplot as plt
import numpy as np
import pickle

# grid World Mode
easy = True

# Create the grid World Env
env = gridWorldEnv(easy)
# Define the Max Epochs and Each Episode Length
maxEpisode = 5000
epLength = 50
# epsilon greedy Exploration Value
epsilon = 0.3
epsilonDecay = 0.999 # Decay parameter for epsilon

# State and Action Dimension
a_dim = env.action_space_size()
s_dim = env.observation_space_size()

# Initialize the Learning Agent
agent = Sarsa(s_dim, a_dim)

# Reward vector for Ploting
epReward = []
avgReward = []

# File Name for saving the Results to file
file_name = 'hw2_sarsa_easyon'


#Start Learning
for epochs in range(maxEpisode):
    state = env.reset()
    total_reward = 0    
    for h in range(epLength):
        action = agent.eGreedyAction(state, epsilon)
        next_state,r,done = env.step(action)
        next_action = agent.eGreedyAction(next_state, 0.0)
        agent.learnQ(state,action,r,next_state, next_action)
        state = next_state
        total_reward += r
    epsilon *= epsilonDecay
    epReward.append(total_reward)

    if epochs % 25 == 0:
        avgReward.append(np.mean(epReward[-25:]))

# Save the results to file
with open(file_name,'wb') as file:
    pickle.dump(avgReward, file) 

# Printing Average Reward for every N epochs  
plt.plot(avgReward)
plt.xlabel('Epochs')
plt.ylabel('Avg Reward')
plt.title('SARSA - GridWorld')
plt.grid(axis='both')
plt.show()