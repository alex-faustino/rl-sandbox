# general packages
import gym
import numpy as np
import matplotlib.pyplot as plt

import gym_gridworld
gw_env = gym.make('GridWorld-v0')

episodes_num = 1
max_time = 100
all_actions = np.array(0)
all_rewards = np.array(0)
for episode in range(episodes_num):
    observation = gw_env.reset()
    episode_total_reward = 0
    for t in range(max_time):
        gw_env.render()
        action = gw_env.action_space.sample()
        all_actions = np.append(all_actions, action)
        observation, reward, done, info = gw_env.step(action)
        all_rewards = np.append(all_rewards, reward)
        episode_total_reward += reward
        if t == max_time - 1:
            print("Total reward earned for episode {0}: {1}".format(episode + 1, episode_total_reward))
            all_rewards = all_rewards[0:-1]
            plt.plot(range(max_time), all_rewards, 'r')
            plt.ylabel('reward')
            plt.xlabel('time index')
            plt.show()
    gw_env.close()

# Remove initializing "action"
all_actions = all_actions[0:-1]
plt.plot(range(max_time), all_actions, 'b')
plt.ylabel('action')
plt.xlabel('time index')
plt.show()

n, bins = np.histogram(all_actions, bins=[0, 1, 2, 3, 4])
print("North was selected {} times".format(n[0]))
print("East was selected {} times".format(n[1]))
print("South was selected {} times".format(n[2]))
print("West was selected {} times".format(n[3]))
