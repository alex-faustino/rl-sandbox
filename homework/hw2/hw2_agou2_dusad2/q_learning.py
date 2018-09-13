from gridworld import GridWorld
import numpy as np
from gym import Env

class Q():
    def __init__(self, default=0):
        self.states_ = {}
        self.default = default

    def get(self, state, action):
        state = str(state)
        if state not in self.states_:
            self.states_[state] = {}
        if action not in self.states_[state]:
            self.states_[state][action] = self.default
        return self.states_[state][action]

    def set(self, state, action, value):
        state = str(state)
        if state not in self.states_:
            self.states_[state] = {}
        self.states_[state][action] = value

    def get_state(self, state):
        state = str(state)
        if state not in self.states_:
            self.states_[state] = {}
        return self.states_[state]

class Q_Learning():
    def __init__(self, env, step_size=0.1, epsilon=0.1, discount=0.9, episode_length=1000):
        self.env = env
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount = discount
        self.episode_length = episode_length
        self.Q = Q()
        

        
    def run_episode(self):
        reward_sum = 0
        state = self.env.reset()
        for _ in range(self.episode_length):
            action = self.epsilon_greedy(state, train=True)
            new_state, reward, _, _ = self.env.step(action)
            best_action = self.always_best_action(new_state)
            reward_sum += reward
            delta = self.step_size*(reward + self.discount*(self.Q.get(new_state, best_action) - self.Q.get(state, action)))
            self.Q.set(state, action, self.Q.get(state, action) + delta)
            state = new_state
        return reward_sum

    def epsilon_greedy(self, state, train=True):
        explore = np.random.choice([True, False], p=[self.epsilon, 1-self.epsilon])
        if train and (explore or len(self.Q.get_state(state).values()) == 0):
            return self.env.action_space.sample()
        max_action = max(self.Q.get_state(state), key=self.Q.get_state(state).get)
        return max_action

    def always_best_action(self, state):
        if len(self.Q.get_state(state).values()) == 0:
            return self.env.action_space.sample()
        max_action = max(self.Q.get_state(state), key=self.Q.get_state(state).get)
        return max_action

    def test(self, steps=10):
        states = []
        state = self.env.reset()
        states.append(state)
        
        for _ in range(steps):
            action = self.epsilon_greedy(state, train=True)
            state, _, _, _ = self.env.step(action)
            states.append(state)
        return states