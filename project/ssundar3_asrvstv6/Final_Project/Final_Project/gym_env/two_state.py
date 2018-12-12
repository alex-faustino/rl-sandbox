import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class two_stateEnv(gym.Env):
    """n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    
    def __init__(self, n=2, slip=0, small=2, large=2):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == self.state:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = self.state
        else:
            reward = 0
            if self.state == 0:
                self.state = 1
            else:
                self.state = 0
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.randint(0, high = 2)
        #print(self.state)
        return self.state
