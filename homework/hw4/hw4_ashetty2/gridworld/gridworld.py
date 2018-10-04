import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class GridWorld(gym.Env):
	metadata = {'render.modes': ['human']}
	action_str = "NSEW"

	GRIDMAP = ["-A-B-",
		   "-----",
		   "---b-",
		   "-----",
		   "-a---"]

	def __init__(self):
		#print('GridWorld loaded')
		self.state = None
		self.level = None

	def step(self, action):
		rown = self.state[0]
		coln = self.state[1]
		done = False
		info = {}
		
		#if at A
		if rown==0 and coln==1:
			new_row = 4
			new_col = 1
			reward = 10
			self.state = np.array([new_row, new_col])
			return self.state, reward, done, info
		#if at B
		elif rown==0 and coln==3:
			new_row = 2
			new_col = 3
			reward = 5
			self.state = np.array([new_row, new_col])
			return self.state, reward, done, info

		new_row, new_col = rown, coln
		reward = 0

		#change action if level is hard
		if self.level=='hard':
			rand_n = np.random.rand(1)[0]
			if rand_n < 0.1:
				action = self.action_str[np.random.randint(4)]

		if action=="N":
			if rown==0:
				reward = -1
			else:
				new_row -= 1
		elif action=="S":
			if rown==4:
				reward = -1
			else:
				new_row += 1
		elif action=="E":
			if coln==4:
				reward = -1
			else:
				new_col += 1
		elif action=="W":
			if coln==0:
				reward = -1
			else:
				new_col -= 1

		self.state = np.array([new_row, new_col])
		return self.state, reward, done, info


	def reset(self, curr_level='easy'):
		self.state = np.random.randint(5, size=2)
		self.level = curr_level
		#print('Generate random initial state:', self.state)	
		return self.state

	def render(self, mode='human', close=False):
		rown = self.state[0]
		coln = self.state[1]
		for row in range(0,5):
			if row == rown:
				temp_str = self.GRIDMAP[row][0:coln] + "X" + self.GRIDMAP[row][coln+1:]
			else:
				temp_str = self.GRIDMAP[row]
			print(temp_str)

	def action_space(self):
		action = np.random.randint(4)
		return self.action_str[action]
