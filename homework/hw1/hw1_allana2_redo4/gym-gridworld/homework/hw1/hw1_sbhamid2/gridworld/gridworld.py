import math
import gym
from gym import spaces, logger, utils
from gym.utils import seeding
import numpy as np
import sys
from six import StringIO, b

GRID = ["00000","00000","00000","00000", "00000"]

'''
Problem-1, Homework-1: 
In this we design a grid world with random actions (left/right/up/down) given as input and 
occasional rewards are obtained based on the current location of the pointer. 
The grid considered is 5x5
Reward =10 is obtained the pointer lands on [0,1] and reward=5 is obtained when pointer lands on [0,3]

'''
class GridWorldEnvNew(gym.Env):

    def __init__(self):

        self.world = 5
        self.a_init = [0, 1]
        self.b_init = [0, 3]
        self.desc = np.asarray(GRID,dtype='c') 
        self.a_prime = [4, 1]
        self.b_prime = [2, 3]
        self.discount = 0.9

        # left, up, right, down
        self.agent_act = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
        self.prob = 0.25

        self.state = None
        self.x = None
        self.y = None
        self.cost = np.zeros((self.world, self.world))

        self.action_space = np.array([spaces.Discrete(2).sample(), spaces.Discrete(2).sample()])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = list(self.np_random.uniform(low=0, high=5, size=(2,)).astype(int) )
        #[spaces.Discrete(self.world).sample(), spaces.Discrete(self.world).sample()]
        return self.state

    def step(self, action):
        state = self.state
        cur_x, cur_y = state
        
        if state == self.a_init:
            next_state = self.a_prime
            x, y = next_state
            reward = 10
        elif state == self.b_init:
            next_state = self.b_prime
            x, y = next_state
            reward = 5
        else:   
            next_state = list(map(sum, zip(state, action))) #(state + action).tolist()
            x, y = next_state
            if x < 0 or x >= self.world or y < 0 or y >= self.world:
                reward = -1.0
                next_state = state
                x, y = next_state
            else:
                reward = 0

        new_cost = self.cost[cur_x, cur_y] + self.prob * (reward + self.discount * self.cost[x, y]) 
        if np.sum(np.abs(self.cost[cur_x, cur_y]- new_cost)) < 1e-4:
            done = 1
        else:
            done = 0
        done = bool(done)
        self.cost[cur_x,cur_y] = new_cost
        self.state = next_state
        self.x = x
        self.y = y
        return self.state, reward, done, {}

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.x, self.y 
        desc = self.desc

        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = '1'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile