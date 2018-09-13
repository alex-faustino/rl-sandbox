import gym
import gym_gridworld
import numpy as np
import matplotlib.pyplot as plt
from tablemethodsrl import Sarsa

LEARNING_RATE = .01  # smaller values learn slower
REWARD_DECAY = .9  # smaller values care more about future rewards
EPS = .1  # small values are greedier

gw_env = gym.make('GridWorld-v0')
sarsa_action_space = list(range(gw_env.action_space.n))
sarsa = Sarsa(sarsa_action_space, LEARNING_RATE, REWARD_DECAY, EPS)

episodes_num = 500
episode_length = 100
all_cum_rewards = np.array(0)
for episode in range(episodes_num):
    observation = gw_env.reset()
    action = sarsa.choose_action(str(observation))
    episode_cum_reward = 0
    for t in range(episode_length):
        gw_env.render()

        observation_plus1, reward, done, info = gw_env.step(action)
        action_plus1 = sarsa.choose_action(str(observation_plus1))
        if t == episode_length - 1:
            observation_plus1 = 'terminal'

        sarsa.learn(str(observation), action, reward, str(observation_plus1), action_plus1)

        observation = observation_plus1
        action = action_plus1

        episode_cum_reward += reward

    all_cum_rewards = np.append(all_cum_rewards, episode_cum_reward)

all_cum_rewards = all_cum_rewards[0:-1]
plt.plot(range(episodes_num), all_cum_rewards, 'r')
plt.ylabel('episode cumulative reward')
plt.xlabel('episode number')
plt.show()
