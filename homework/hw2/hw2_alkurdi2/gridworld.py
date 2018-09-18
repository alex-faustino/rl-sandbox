"""
"""

import math
import gym
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np

class GridworldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,hard_version=False):
        self.hard_version = hard_version

        
        
        #since it's a 5x5 grid and x&y are xy coordinates of each block
        self.x_max = 5
        self.y_max = 5
        
        
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
        if hard_version=True 
            if random.random() > 0.9:
                pass
            else:
                action=random.randint(1,4)
            
            
        if x==2 and y==5 :
            reward=10
            self.state=[2,1]
        elif x==4 and y==5:
            reward=10
            self.state=[4,3]
        elif (x==1 and action==2) or (x==5 and action==1) or (y==1 and action==4) or (y==5 and action==3):
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
