# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 10:52:56 2018

@author: sreen
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib as plt
from pylab import *
#import matplotlib.pyplot as plt

logger = logging.getLogger(__name__) 

#global Ar,Ac,A_r,A_c,Br,Bc,B_r,B_c,low,high
global A,A_,B, B_, low, high
A = [4,1]
A_ = [0,1]
B = [4,3]
B_ = [2,3]
#Ar = 4
#Ac = 1
#A_r = 0
#A_c = 1
#Br = 4
#Bc = 3
#B_r = 2
#B_c = 3
low = 0
high = 4
rangelist = range(0,5)
print(A)
print(A_)
print(B)

class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1
                }
    def __init__(self):
        grid_map = np.zeros((5,5))
        grid_dim_x,grid_dim_y = grid_map.shape
        self.actionspace = spaces.Discrete(4)
        #self.actioncoord = {'0': [1,0],'1':[-1,0],'2':[0,1], '3':[0,-1]}
        #print(self.actioncoord[0][0])
        self.actioncoord = [[1,0],[-1,0],[0,1],[0,-1]]
        self._seed()
        self.viewer = None
        self.state = None
        self.state = list(np.random.randint(low,high,size=(2,)))
        print(self.state)
        
        
    def _seed(self, seed = None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]
    
    def step(self,action):
        #assert self.action_space.contains(action),"%r (%s) invalid"%(action,type(action))
        #global reward, Tset, low, high, Twi, Tai, fa
        global A,A_,B, B_, low, high,rangelist
        action =  int(action)
        print(action)
        if (self.state == A):
            reward = 10
            self.state = A_
        elif (self.state==B):
            reward = 5
            self.state = B_
        else:
            new_agent_state = [self.state[0]+self.actioncoord[action][0],self.state[1]+self.actioncoord[action][1]]
            if (new_agent_state[0] in rangelist) and (new_agent_state[1] in rangelist):
                self.state = new_agent_state
                reward = 0
                pass
            else:
                reward = -1
        return self.state,reward,False,{}
    
    def reset(self):
        global low, high
        self.state = list(np.random.randint(low,high,size=(2,)))
        print(self.state)
        #self.steps_beyond_done = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500
        fac = 100
        x,y = self.state
        x+=1
        y+=1
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            for i in range(6):
                line_start_x = (0,i*fac)
                line_end_x = (5*fac,i*fac)
                line_start_y = (i*fac,0)
                line_end_y = (i*fac,5*fac)
                line1 = self.viewer.draw_line(line_start_x,line_end_x)
                self.viewer.add_geom(line1)
                line2 = self.viewer.draw_line(line_start_y,line_end_y)
                self.viewer.add_geom(line2)
            box_size = 30
            left, right, top, bottom = -box_size, +box_size, +box_size, -box_size
            box = rendering.FilledPolygon([(left, bottom), (left, top), (right, top), (right, bottom)])
            self.box_trans = rendering.Transform()
            box.add_attr(self.box_trans)
            box.set_color(.8, .4, .4)
            self.viewer.add_geom(box)

        if self.state is None:
            return None

        self.box_trans.set_translation((x-0.5)*100, (y-0.5)*100)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


env = GridWorldEnv()
reward = 0

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print('Observation : [%s], Reward: %d'%(','.join(map(str,observation)),reward))
        action = env.actionspace.sample()
        observation, reward, done, info = env.step(action)

