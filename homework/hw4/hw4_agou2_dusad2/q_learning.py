import numpy as np
from gym import Env
from tqdm import tqdm_notebook

class Q_Learning():
    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((25, 4))
        
        
    def run_episode(self, step_size=0.1, epsilon=0.1, discount=0.9, episode_length=1000):
        rewards = np.zeros(episode_length)
        state = self.env.reset()
        for i in range(episode_length):
            action = self.epsilon_greedy(state, epsilon=epsilon)
            new_state, reward, _, _ = self.env.step(action)
            best_action = self.epsilon_greedy(new_state, epsilon=0)
            rewards[i] = reward
            delta = step_size*(reward + discount*(self.Q[new_state, best_action] - self.Q[state, action]))
            self.Q[state, action] += delta
            state = new_state
        return rewards.sum()

    def train(self, num_episodes=1000, step_size=0.1, epsilon=0.1, discount=0.9, episode_length=100, decay=None):
        rewards = np.zeros(num_episodes)
        for i in tqdm_notebook(range(num_episodes)):
            if decay is None:
                rewards[i] = self.run_episode(step_size, epsilon, discount, episode_length)
            else:
                rewards[i] = self.run_episode(step_size, epsilon*np.exp(-decay*i), discount, episode_length)
        return rewards


    def epsilon_greedy(self, state, epsilon):
        explore = np.random.choice([True, False], p=[epsilon, 1-epsilon])
        if explore:
            return np.random.randint(self.env.action_space.n)
        max_action = self.Q[state,:].argmax()
        return max_action

    def test(self, steps=10):
        states = []
        state = self.env.reset()
        states.append(state)
        
        for _ in range(steps):
            action = self.epsilon_greedy(state, epsilon=0)
            state, _, _, _ = self.env.step(action)
            states.append(state)
        return states