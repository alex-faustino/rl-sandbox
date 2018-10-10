from gym import Env, logger, spaces
from gym.utils import seeding, colorize
import numpy as np


class GridWorldEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.xdim = 5
        self.ydim = 5

        self.moves = {'north':0,'south':1,'west':2,'east':3}

        self.action_space = spaces.Discrete(len(self.moves))
        self.observation_space = spaces.Discrete(self.xdim * self.ydim)
        self.seed()
        self.reset()
        self.stochastic_transitions = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        map = np.zeros((self.xdim, self.ydim))
        map[self.pointA[1]][self.pointA[0]] = 1
        map[self.pointAprime[1]][self.pointAprime[0]] = 2
        map[self.pointB[1]][self.pointB[0]] = 3
        map[self.pointBprime[1]][self.pointBprime[0]] = 4
        map[self.location[1]][self.location[0]] = 8

        print(map)
        return None

    def step(self, action):
        # assert self.action_space.contains(action)

        reward = 0

        # stochastic transitions
        if self.stochastic_transitions and self.np_random.rand() < 0.1:
            action = self.np_random.randint(len(self.moves))

        # termination steps
        if np.array_equal(self.location,self.pointA):
            self.location = np.copy(self.pointAprime)
            reward = 10
        elif np.array_equal(self.location,self.pointB):
            self.location = np.copy(self.pointBprime)
            reward = 5
        #all other steps
        elif action == self.moves['north'] and self.location[1] > 0:
             self.location[1] = self.location[1] - 1
        elif action == self.moves['south'] and self.location[1] < self.ydim - 1:
             self.location[1] = self.location[1] + 1
        elif action == self.moves['east'] and self.location[0] > 0:
             self.location[0] = self.location[0] - 1
        elif action == self.moves['west'] and self.location[0] < self.xdim - 1:
             self.location[0] = self.location[0] + 1
        else:
            reward = -1

        done = False

        return (self.location, reward, done, {})

    def reset(self, A=[1,0], Ap=[1,4], B=[3,0], Bp=[3,2]):

        self.pointA = A # x,y loc
        self.pointAprime = Ap

        self.pointB = B
        self.pointBprime = Bp

        self.location = [self.np_random.randint(self.xdim),
                        self.np_random.randint(self.ydim)]

        return np.copy(self.location)
