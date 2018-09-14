import gym
import gym_gridworld
import numpy as np
import matplotlib.pyplot as plt
from tablemethodsrl import Sarsa

LEARNING_RATE = .01  # smaller values learn slower
REWARD_DECAY = .9  # smaller values care more about future rewards
EPS = .1  # smaller values are greedier

gw_env = gym.make('GridWorld-v0')
agent = Sarsa(gw_env, LEARNING_RATE, REWARD_DECAY, EPS)

episodes_num = 500
episode_length = 100
all_cum_rewards = np.array(0)
for episode in range(episodes_num):
    # initialize each episode
    s = gw_env.reset()
    a = agent.choose_action(s)

    episode_cum_reward = agent.train(s, a, episode_length, False)

    all_cum_rewards = np.append(all_cum_rewards, episode_cum_reward)

all_cum_rewards = all_cum_rewards[0:-1]
plt.plot(range(episodes_num), all_cum_rewards, 'r')
plt.ylabel('episode cumulative reward')
plt.xlabel('episode number')
plt.show()
