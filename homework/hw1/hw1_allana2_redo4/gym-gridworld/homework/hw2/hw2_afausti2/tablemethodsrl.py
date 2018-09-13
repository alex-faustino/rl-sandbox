import numpy as np
import pandas as pd

# Pretty standard implementations of SARSA and Q-Learning in Python
# Adapted from: github.com/MorvanZhou/Reinforcement-learning-with-tensorflow


class GenTableMethodsRL(object):
    def __init__(self, action_space, learning_rate, reward_decay, eps):
        self.actions = action_space
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.eps = eps

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_in_table(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state,)
            )

    def choose_action(self, observation):
        self.check_state_in_table(observation)
        # action selection
        if np.random.rand() < (1 - self.eps):
            # exploitation
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose one in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # exploration
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class Sarsa(GenTableMethodsRL):
    def __init__(self, actions, learning_rate, reward_decay, eps):
        super(Sarsa, self).__init__(actions, learning_rate, reward_decay, eps)

    def learn(self, s, a, r, s_plus1, a_plus1):
        self.check_state_in_table(s_plus1)
        q_predict = self.q_table.loc[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table.loc[s_plus1, a_plus1]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate*(q_target - q_predict)


class QLearning(GenTableMethodsRL):
    def __init__(self, actions, learning_rate, reward_decay, eps):
        super(QLearning, self).__init__(actions, learning_rate, reward_decay, eps)

    def learn(self, s, a, r, s_plus1):
        self.check_state_in_table(s_plus1)
        q_predict = self.q_table.loc[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table.loc[s_plus1, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)
