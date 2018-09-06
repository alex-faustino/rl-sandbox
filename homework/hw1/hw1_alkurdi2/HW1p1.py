"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Gridworldaa(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
#        self.gravity = 9.8
#        self.masscart = 1.0
#        self.masspole = 0.1
#        self.total_mass = (self.masspole + self.masscart)
#        self.length = 0.5 # actually half the pole's length
#        self.polemass_length = (self.masspole * self.length)
#        self.force_mag = 10.0
#        self.tau = 0.02  # seconds between state updates
#
#        # Angle at which to fail the episode
#        self.theta_threshold_radians = 12 * 2 * math.pi / 360
#        self.x_threshold = 2.4
        
        
        
        #since it's a 5x5 grid and x&y are xy coordinates of each block
        self.x_max = 5
        self.y_max = 5
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
#        high = np.array([
#            self.x_threshold * 2,
#            np.finfo(np.float32).max,
#            self.theta_threshold_radians * 2,
#            np.finfo(np.float32).max])
        high = np.array([self.x_max,self.y_max])
        low = np.array([1,1])
        
        
        
        # 1 right 2 left
        # 3 up    4 down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high,shape=[5,5])

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

   
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, y = state
        
        self.state = (x,y)
        if random.random() > 0.9:
            pass
        else:
            action=random.randint(1,4)
            
            
        if x == 2 && y == 5:
            reward=10
            self.state=[2,1]
        elif x==4 && y==5:
            reward=10
            self.state=[4,3]
        elif (x==1 && action==2) || (x==5 && action==1) || (y==1 && action==4) || (y==5 && action==3):
            reward=-1
            self.state=[x,y]
        else:
            reward=0
            if action ==1:
                x=x+1
            if action ==2:
                x=x-1
            if action ==3:
                y=y+1
            if action ==4:
                y=y-1
            self.state=[x,y]
        done= False 
        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = self.np_random.uniform(low=1, high=5, size=(2,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        pass
        
    def close(self):
        if self.viewer: self.viewer.close()
