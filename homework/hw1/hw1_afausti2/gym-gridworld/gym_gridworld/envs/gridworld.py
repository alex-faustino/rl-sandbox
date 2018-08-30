import os, sys, time
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class GridWorldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		...
	def _step(self, action):
		...
	def _reset(self):
		...
	def _render(self, mode='human', close=False):
		...
	
	def _take_action(self, action):
		pass

	def _get_reward(self):
		""" Reward is given for XY. """
        if self.status == FOOBAR:
			return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0
