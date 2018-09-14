import numpy as np

# Pretty standard implementations of SARSA and Q-Learning in Python
# Adapted from: github.com/MorvanZhou/Reinforcement-learning-with-tensorflow


class GenTableMethodsRL(object):
    def __init__(self, env, learning_rate, reward_decay, eps):
        self.env = env
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.eps = eps

        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, s):
        if np.random.rand() < (1 - self.eps):
            # exploitation
            a = self.q_table[s, :].argmax()
        else:
            # exploration
            a = self.env.action_space.sample()
        return a

    def train(self, *args):
        pass

    def learn(self, *args):
        pass


class Sarsa(GenTableMethodsRL):
    def __init__(self, env, learning_rate, reward_decay, eps):
        super(Sarsa, self).__init__(env, learning_rate, reward_decay, eps)

    def train(self, s, a, max_episode_length, viz):
        cum_reward = 0
        for t in range(max_episode_length):
            if viz:
                self.env.render()

            s_plus1, r, done, info = self.env.step(a)
            a_plus1 = self.choose_action(s_plus1)
            if t == max_episode_length - 1:  # temporary way to end each episode cleanly
                s_plus1 = 'terminal'

            # update q
            self.learn(s, a, r, s_plus1, a_plus1)

            # cycle
            s = s_plus1
            a = a_plus1

            # accumulate reward
            cum_reward += r

        return cum_reward

    def learn(self, s, a, r, s_plus1, a_plus1):
        q_predict = self.q_table[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table[s_plus1, a_plus1]
        else:
            q_target = r
        self.q_table[s, a] += self.learning_rate*(q_target - q_predict)


class QLearning(GenTableMethodsRL):
    def __init__(self, env, learning_rate, reward_decay, eps):
        super(QLearning, self).__init__(env, learning_rate, reward_decay, eps)

    def train(self, s, max_episode_length, viz):
        cum_reward = 0
        for t in range(max_episode_length):
            if viz:
                self.env.render()

            a = self.choose_action(s)
            s_plus1, r, done, info = self.env.step(a)
            if t == max_episode_length - 1:  # temporary way to end each episode cleanly
                s_plus1 = 'terminal'

            # update q
            self.learn(s, a, r, s_plus1)

            # cycle
            s = s_plus1

            # accumulate reward
            cum_reward += r

        return cum_reward

    def learn(self, s, a, r, s_plus1):
        q_predict = self.q_table[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table[s_plus1, :].max()
        else:
            q_target = r
        self.q_table[s, a] += self.learning_rate * (q_target - q_predict)
