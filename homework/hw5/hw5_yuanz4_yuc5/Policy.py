import numpy as np

class Policy():
    def __init__(self, level, epsilon):
        # level = 'easy' or 'hard'
        self.level = level
        self.epsilon = epsilon
        # left, up, right, down
        self.ACTIONS = [np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1]),
                        np.array([1, 0])]
        self.WORLD_SIZE = 5
        self.PROB = 10
        self.A_POS = np.array([0, 1])
        self.A_PRIME_POS = np.array([4, 1])
        self.B_POS = np.array([0, 3])
        self.B_PRIME_POS = np.array([2, 3])
        self.policy = np.zeros((self.WORLD_SIZE**2, 4))

    def action(self):
        if self.level == 'hard':
            if np.random.rand() < self.epsilon:
                return np.random.randint(4)
        x, y = self.state
        s = x * self.WORLD_SIZE + y
        p_actions = np.exp(self.policy[s]) / np.sum(np.exp(self.policy[s]))
        a = np.random.choice(len(p_actions), 1, p=p_actions)
        return int(a)

    def step(self, action):
        done = self.done
        act = self.ACTIONS[action]
        next_state = self.state + act
        x, y = self.state
        reward = 0
        if (next_state == self.A_POS).all():
            next_state = self.A_PRIME_POS
            reward = 10
        if (next_state == self.B_POS).all():
            next_state = self.B_PRIME_POS
            reward = 5
        x_, y_ = next_state
        if x_ < 0 or x_ >= self.WORLD_SIZE or y_ < 0 or y_ >= self.WORLD_SIZE:
            reward = -1
        else:
            self.state = next_state
        return self.state, reward, done, {}

    def reset(self):
        self.done = 0
        self.state = np.array(np.random.randint(self.WORLD_SIZE, size=2))
        return self.state

    def update(self, policy):
        self.policy += policy

    def get_policy(self):
        return self.policy
