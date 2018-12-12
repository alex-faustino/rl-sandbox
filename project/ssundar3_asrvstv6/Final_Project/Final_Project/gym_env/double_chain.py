import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class double_chainEnv(gym.Env):    

    def __init__(self, n=9, slip=0, small=2, large=10):
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
        
        #if action == 0 and (self.state !=4 and self.state != 8 and self.state != 0):
        if action == 0 and (self.state !=4 and self.state != 8 and self.state != 0):
            reward = 0
            self.state = self.state + 1
        elif action == 0 and self.state == 4:
            reward = 10
            self.state = 4
        elif action == 0 and self.state == 8:
            reward = 5
            self.state = 8
        elif action == 0 and self.state == 0:
            reward = 0
            self.state = self.state + 1
        elif action == 1 and self.state == 0:
            reward = 0
            self.state = 5
        else:
            reward = 2
            self.state = 0
            
        done = False
        return self.state, reward, done, {}

    def reset(self, trip_value = 8):
        #self.state = np.random.randint(0, high = 9)
        self.state = 0
        #self.state = np.random.randint(0, high = 9)
        #print(self.state)
        return self.state