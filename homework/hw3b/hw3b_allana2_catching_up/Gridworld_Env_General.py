import numpy as np
import matplotlib.pyplot as plt
import pylab
import gym
import sys
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.ticker as plticker

class gridworld(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self): 
#    self.render_label = ''
    self.gridnum = int(7) #size of gridworld
    self.location_x = np.random.randint(int(1),self.gridnum-1)
    self.location_y = np.random.randint(int(1),self.gridnum-1)
    self.action = int(0) # no initial action until computed
    self.allowed_actions = np.array([1,2,3,4])[np.newaxis]
    self.previous_action = self.action
    self.previous_x = self.location_x
    self.previous_y = self.location_y
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.allowed_actions = np.array([1,2,3,4])[np.newaxis]
    self.my_reward =np.array([ [-1,-1,-1,-1,-1,-1,-1],[-1, 0, 10, 0, 5, 0, -1],[-1, 0, 0, 0, 0, 0, -1],\
    [-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, -1, -1, -1, -1, -1, -1] ]) 
    self.my_reward = np.flipud(self.my_reward)
    self.color_array = ['blue','orange']
    self.episode_counter = 0
    self.my_exploit_action_log = np.random.rand(self.gridnum,self.gridnum)
    pass
  def states_and_actions(self):
    return self.gridnum**2, self.allowed_actions
  def render(self,fig,ax,time_index,render_label): #mode='human', close=False <- no idea what this is for
    ax = fig.gca()    
    ax.clear()
    ax.grid(which='major', axis='both', linestyle='-')
    circle2 = plt.Circle((self.location_x+0.5, self.location_y+0.5), 0.5, color=self.color_array[np.mod(self.episode_counter,2)])#rand initialization
    ax.add_artist(circle2)
    fig.canvas.draw()
    pass
  def render_init(self,render_label):
    fig, (ax)=plt.subplots()
    intervals = float(1/self.gridnum)# dimension of grid affects size
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_major_locator(loc)
    ax.set_xlim(0, self.gridnum)
    ax.yaxis.set_major_locator(loc)
    ax.set_ylim(0, self.gridnum)
    return fig, ax
  def reset(self):
    self.location_x = np.random.randint(int(1),self.gridnum-1)
    self.location_y = np.random.randint(int(1),self.gridnum-1)
    self.previous_x = self.location_x
    self.previous_y = self.location_y
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.update_q_label = 0
    self.episode_counter += 1
    return self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y, self.update_q_label, self.episode_counter
  def step(self, my_reward, desired_action, location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y):
    (self.my_reward, self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y) = (my_reward, location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y)
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.previous_x = self.location_x# only accurate for second step in episode
    self.previous_y = self.location_y# only accurate for second step in episode
    self.action = desired_action
    if np.random.rand() <= 0.1: #10% chance of a random transition
        self.action = self.allowed_actions[0,np.random.randint(0,self.allowed_actions.shape[1])]
    if self.my_reward[self.location_y,self.location_x] > 5:
        self.location_x = 1+1
        self.location_y = 0+1
    elif self.my_reward[self.location_y,self.location_x] > 0:
        self.location_x = 3+1
        self.location_y = 2+1
    elif self.action == 1: # this part of the method is to select the desired deterministic action
        self.location_y += 1
    elif self.action == 2:
        self.location_y += -1
    elif self.action == 3:
        self.location_x += -1
    elif self.action == 4:
        self.location_x += 1
    return (self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y)
