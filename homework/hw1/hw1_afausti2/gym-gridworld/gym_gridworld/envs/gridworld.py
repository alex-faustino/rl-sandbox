import os, sys, time
import gym
import numpy as np
from gym import error, spaces, utils

class GridWorldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		'''' set action space ''''
		'''' 0 = North ''''
		'''' 1 = East ''''
		'''' 2 = South ''''
		'''' 3 = West ''''
		self.action_space = spaces.Discrete(4)
		
		'''' set observation space ''''
		self.obs_low = -2
		self.obs_high = 2
		self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, shape=[5, 5])
	
		self.state = None
		
	def _step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x1, x2 = state
        
        '''' handle special cases ''''
        if x1 == -1 && x2 == 2:
			reward = 10
			self.state = [-1, -2]
		elif x1 == 1 && x2 == 2:
			reward = 5
			self.state = [1, 0]
        '''' handle edge cases ''''
        elif (x1 == -2 && action == 3) || (x1 == 2 && action == 1) || (x2 == -2 && action == 2) || (x2 == 2 && action == 0):
			reward = -1
			self.state = [x1, x2]
		'''' normal cases ''''
		else
			reward = 0
			if action == 0: x2 += 1
			if action == 1: x1 += 1
			if action == 2: x2 += -1
			if action == 3: x1 += -1
			self.state = [x1, x2]
			
		done = False
				
        return np.array(self.state), reward, done, {}
	
	def _reset(self):
		self.state = self.np_random.randint(low=-2, high=2, size=(2,))
		self.steps_beyond_done = None
		return np.array(self.state)
				
	def _render(self, mode='human', close=False):
		pass
		
		
	
