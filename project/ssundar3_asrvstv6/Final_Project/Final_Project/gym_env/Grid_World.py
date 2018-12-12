# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:48:58 2018

@author: sreen
"""
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import time
import numpy as np
import matplotlib as plt
from pylab import *
#import matplotlib.pyplot as plt

logger = logging.getLogger(__name__) 

global A,A_,B, B_, low, high
A = (4,1)
A_ = (0,1)
B = (4,3)
B_ = (2,3)
rangelist = range(0,5)


class Grid(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1
                }
    def __init__(self):
        grid_map = np.zeros((5,5))
        grid_dim_x,grid_dim_y = grid_map.shape
        self.action_space = spaces.Discrete(4)
        #self.state_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(25)
        self.actioncoord = [(1,0),(-1,0),(0,1),(0,-1)]
        self._seed()
        self.viewer = None
        self.state = None
        self.state = np.random.randint(0, high = 25)
        #print(self.state)
        
    def _seed(self, seed = None):
        self.np_random,seed = seeding.np_random(int(time.time()))
        return [seed]
    
    def step(self,action):
        #assert self.action_space.contains(action),"%r (%s) invalid"%(action,type(action))
        #global reward, Tset, low, high, Twi, Tai, fa
        d = {}
        d_inv = {}
        k = 0
        for i in range(np.int(np.sqrt(self.observation_space.n))):
            for j in range(np.int(np.sqrt(self.observation_space.n))):
                d.update([(k, (i,j))])
                d_inv.update([((i,j),k)])
                k+=1
                
        s = d[self.state]
        global A,A_,B, B_, low, high,rangelist
        action =  int(action)
        #print(action)
        if (s == A):
            reward = 10
            self.state = d_inv[A_]
        elif (s == B):
            reward = 5
            self.state = d_inv[B_]
        else:
            new_agent_state = (s[0]+self.actioncoord[action][0],s[1]+self.actioncoord[action][1])
            if (new_agent_state[0] in rangelist) and (new_agent_state[1] in rangelist):
                self.state = d_inv[new_agent_state]
                reward = 0
                pass
            else:
                reward = -1
        return self.state,reward,False,{}
    
    def reset(self):
        global low, high
        self.state = np.random.randint(0, high = 25)
        #list(np.random.randint(low,high,size=(2,)))
        #print(self.state)
        #self.steps_beyond_done = 0
        self._seed()
        return self.state
    
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
     
    