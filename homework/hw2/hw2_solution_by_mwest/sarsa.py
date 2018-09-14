
import random
import numpy as np

class SarsaAgent(object):

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def train(self, epsilon, alpha, gamma, episode_max_length, num_episodes):
        self.reset()
        rewards = np.zeros(num_episodes)
        for i_episode in range(num_episodes):
            s = self.env.reset()
            a = self.action_epsilon_greedy(s, epsilon)
            for i_step in range(episode_max_length):
                (s_new, r, done) = self.env.step(a)
                rewards[i_episode] += r
                a_new = self.action_epsilon_greedy(s_new, epsilon)
                self.Q[s,a] += alpha*(r + gamma*self.Q[s_new,a_new] - self.Q[s,a])
                (s, a) = (s_new, a_new)
                if done:
                    break
        return rewards
            
    def action_greedy(self, s):
        return self.Q[s,:].argmax()
            
    def action_uniform_random(self, s):
        return random.randrange(self.env.action_space.n)
            
    def action_epsilon_greedy(self, s, epsilon):
        if random.random() < epsilon:
            return self.action_uniform_random(s)
        return self.action_greedy(s)

    def enjoy(self, episode_max_length):
        rewards = np.zeros(episode_max_length)
        states = np.zeros(episode_max_length, dtype=int)
        actions = np.zeros(episode_max_length, dtype=int)
        s = self.env.reset()
        for i_step in range(episode_max_length):
            a = self.action_greedy(s)
            states[i_step] = s
            actions[i_step] = a
            (s_new, r, done) = self.env.step(a)
            rewards[i_step] = r
            s = s_new
            if done:
                break
        return (rewards, states, actions)
