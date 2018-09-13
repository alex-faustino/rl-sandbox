import numpy as np

class Q:

    def __init__(self, s, a):

        self.state_num = s
        self.action_num = a
        self.action = None
        self.Q_value = np.zeros((s, a))
        self.alpha = np.ones((s, a))
        self.gamma = 0.95
        self.epsilon = 0.1

    def update(self, a, s, s_prime, r):

        self.Q_value[s][a] = self.Q_value[s][a] + \
        (r + self.gamma*np.amax(self.Q_value[s_prime][:]) \
        -self.Q_value[s][a])*(np.power(1/self.alpha[s][a],0.8))

        self.alpha[s][a] = self.alpha[s][a] + 1

    def greedy(self,s):

        greed_sel = np.random.random_sample()

        if greed_sel >= self.epsilon:
            action_rand = np.argmax(self.Q_value[s][:])
        else:
            action_rand = np.random.randint(low = 0, high = self.action_num)

        return action_rand

class SARSA:

    def __init__(self, s, a):

        self.state_num = s
        self.action_num = a
        self.action = None
        self.Q_value = np.zeros((s, a))
        self.alpha = np.ones((s, a))
        self.gamma = 0.95
        self.epsilon = 0.1

    def update(self, a,a_prime, s, s_prime, r):

        self.Q_value[s][a] = self.Q_value[s][a] + \
        (r + self.gamma*self.Q_value[s_prime][a_prime] \
        -self.Q_value[s][a])*(np.power(1/self.alpha[s][a],0.8))

        self.alpha[s][a] = self.alpha[s][a] + 1

    def greedy(self,s):

        greed_sel = np.random.random_sample()

        if greed_sel >= self.epsilon:
            action_rand = np.argmax(self.Q_value[s][:])
        else:
            action_rand = np.random.randint(low = 0, high = self.action_num)

        return action_rand
                
