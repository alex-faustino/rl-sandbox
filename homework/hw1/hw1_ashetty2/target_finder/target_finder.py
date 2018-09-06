import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TargetFinder(gym.Env):
	metadata = {'render.modes': ['human']}

	GRID_SIZE = 20
	N_TARGETS = 4

	OBS_RANGE = 4
	RENDER_OBS = True

	def __init__(self):
		self.state = None
		self.gridmap = None
		self.obsmap = None
		self.target_locations = None
		self.goal = None

	def step(self, action):
		rown = self.state[0]
		coln = self.state[1]
		reward = 0
		done = False
		info = {}
		
		new_row, new_col = rown, coln
		if action=="N":
			if rown==0:
				reward = -1
			else:
				new_row -= 1
		elif action=="S":
			if rown==self.GRID_SIZE-1:
				reward = -1
			else:
				new_row += 1
		elif action=="E":
			if coln==self.GRID_SIZE-1:
				reward = -1
			else:
				new_col += 1
		elif action=="W":
			if coln==0:
				reward = -1
			else:
				new_col -= 1

		#update state
		self.state = np.array([new_row, new_col])

		#check if new position is goal position in gridmap
		if self.gridmap[new_row][new_col] == self.goal:
			done = True
			reward = 10

		#update gridmap for new state
		self.update_gridmap()

		#update obsmap for new state
		self.update_obsmap()

		return self.obsmap, reward, done, info

	def update_gridmap(self):
		
		target_str = ''
		for i in range(0, self.N_TARGETS):
			target_str += chr(65+i)	

		tmp_str = ''
		tmp_map = []

		for i in range(0, self.GRID_SIZE*self.GRID_SIZE):
			rn = int ( i / self.GRID_SIZE )
			cn = int ( i % self.GRID_SIZE )
			
			if i in self.target_locations:
				tmp_str += target_str[np.where(self.target_locations==i)[0][0]]
			else:
				tmp_str += '.'

			if cn == self.GRID_SIZE-1:
				tmp_map.append(tmp_str)
				tmp_str = ''

		rown = self.state[0]
		coln = self.state[1]

		new_map = []
		for row in range(0,self.GRID_SIZE):
			if row == rown:
				tmp_str = tmp_map[row][0:coln] + "X" + tmp_map[row][coln+1:]
			else:
				tmp_str = tmp_map[row]
			
			new_map.append(tmp_str)

		self.gridmap = new_map

	def update_obsmap(self):
		
		rown = self.state[0]
		coln = self.state[1]
			
		obs_map = []
		for rn in range(rown-self.OBS_RANGE, rown+self.OBS_RANGE+1):
			
			if rn<0 or rn>=self.GRID_SIZE:
				tmp_str = '*'*(2*self.OBS_RANGE+1)
			else:
				tmp_str = '' 
				for cn in range(coln - self.OBS_RANGE, coln + self.OBS_RANGE+1):
					if cn<0 or cn>=self.GRID_SIZE:
						tmp_str += '*'	
					else:
						tmp_str += self.gridmap[rn][cn]
			obs_map.append(tmp_str)
		self.obsmap = obs_map

	def reset(self, goal_target="A"):

		imp_points = np.random.choice(self.GRID_SIZE*self.GRID_SIZE, self.N_TARGETS+1, replace=False)
		state_rn = int ( imp_points[-1] / self.GRID_SIZE )
		state_cn = int ( imp_points[-1] % self.GRID_SIZE )

		self.state = np.array([ state_rn, state_cn ])
		self.target_locations = imp_points[:-1]
		self.goal = goal_target

		self.update_gridmap()
		self.update_obsmap()

		return self.obsmap

	def render(self, mode='human', close=False):

		for i in range(0, self.GRID_SIZE):
			print(self.gridmap[i])

		if self.RENDER_OBS==True:
			print('\nObservation:\n')
			for i in range(0, 2*self.OBS_RANGE+1):
				print(self.obsmap[i])


	def action_space(self):
		action = np.random.randint(4)
		action_str = "NSEW"
		return action_str[action]
