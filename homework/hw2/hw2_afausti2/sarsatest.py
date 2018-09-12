import gym
import gym_gridworld
from tablemethodsrl import Sarsa


LEARNING_RATE = .01
REWARD_DECAY = .9
EPS = .1

gw_env = gym.make('GridWorld-v0')
sarsa_action_space = list(range(gw_env.action_space.n))
sarsa = Sarsa(sarsa_action_space, LEARNING_RATE, REWARD_DECAY, EPS)

episodes_num = 40
max_time = 1000
for episode in range(episodes_num):
    observation = gw_env.reset()
    action = sarsa.choose_action(str(observation))
    episode_total_reward = 0
    for t in range(max_time):
        # gw_env.render()
        # action = gw_env.action_space.sample()
        # all_actions = np.append(all_actions, action)

        observation_plus1, reward, done, info = gw_env.step(action)
        action_plus1 = sarsa.choose_action(str(observation_plus1))

        sarsa.learn(str(observation), action, reward, str(observation_plus1), action_plus1)

        observation = observation_plus1
        action = action_plus1

        # all_rewards = np.append(all_rewards, reward)
        episode_total_reward += reward
        # if t == max_time - 1:
        #     print("Total reward earned for episode {0}: {1}".format(episode + 1, episode_total_reward))
        #     all_rewards = all_rewards[0:-1]
        #     plt.plot(range(max_time), all_rewards, 'r')
        #     plt.ylabel('reward')
        #     plt.xlabel('time index')
        #     plt.show()
    # gw_env.close()
    print(episode_total_reward)
