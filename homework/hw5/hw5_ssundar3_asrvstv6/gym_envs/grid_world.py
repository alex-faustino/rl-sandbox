'''
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Grid(gym.Env):            #deriving class Grid from base class gym.Env
    metadata = {
            'render.modes': ['human']
            }
    
    def __init__(self):        #constructor for the class Grid
        
        #specifying positions of A,B,A' and B'
        self.Ax = 2
        self.Ay = 5
        self.Bx = 4
        
        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Discrete(5)
        
        self.By = 5
        self.Ax_p = 2
        self.Ay_p = 1 
        self.Bx_p = 4
        self.By_p = 3
        self.seed()
        self.viewer = None
        self.state = None
        
        self.steps_beyond_done = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" %(action, type(action))
        state = self.state
        x = state[0] 
        y = state[1]
        x_old = state[0]
        y_old = state[1]
        done = False
        
        if x_old == self.Ax and y_old == self.Ay:
            x = self.Ax_p
            y = self.Ay_p
            reward = 10
        elif x_old == self.Bx and y_old == self.By:
            x = self.Bx_p
            y = self.By_p
            reward = 5
        elif action == 0:
            x = x_old
            y = y_old + 1
            reward = 0
        elif action == 1:
            x = x_old + 1
            y = y_old
            reward = 0
        elif action == 2:
            x = x_old
            y = y_old - 1
            reward = 0
        elif action == 3:
            x = x_old - 1
            y = y_old
            reward = 0
            
        if x > 5 or x < 1 or y > 5 or y < 1:
            x = x_old
            y = y_old
            reward = -1
            
        self.state = (x, y)
        
        return np.array(self.state), reward, done, {}
    
    def reset(self):
        #self.state = (env.state_space.sample()+1, env.state_space.sample()+1)
        self.state = (np.random.randint(1,5,size=(2,)))
        return np.array(self.state)
    
    def render(self, mode = 'human'):
        width = 250
        height = 250
        scale = 50
        state = self.state
        x, y = state
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width, height)
            for i in range(6):
                p1_y = i * scale
                p1_x = 0
                p2_y = i * scale
                p2_x = 5 * scale
                l = self.viewer.draw_line((p1_x,p1_y), (p2_x, p2_y))
                self.viewer.add_geom(l)
            for i in range(6):
                p1_y = 0
                p1_x = i * scale
                p2_y = 5 * scale
                p2_x = i * scale
                l = self.viewer.draw_line((p1_x,p1_y), (p2_x, p2_y))
                self.viewer.add_geom(l)
            
            agent_size = 20
            left = -agent_size
            right = agent_size
            top = agent_size
            bottom = -agent_size
            
            agent = rendering.FilledPolygon([(left, bottom), (left,top), (right,top), (right, bottom)])
            self.agent_trans = rendering.Transform()
            agent.add_attr(self.agent_trans)
            agent.set_color(0,0,0)
            self.viewer.add_geom(agent)
            
        if self.state is None:
            return None
        
        self.agent_trans.set_translation((x-0.5)*50, (y-0.5)*50)
        
        return self.viewer.render(return_rgb_array=mode == 'rbg_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
'''

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
import time
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
#print(A)
#print(A_)
#print(B)

class Grid(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1
                }
    def __init__(self):
        grid_map = np.zeros((5,5))
        grid_dim_x,grid_dim_y = grid_map.shape
        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Discrete(5)
        #self.actioncoord = {'0': [1,0],'1':[-1,0],'2':[0,1], '3':[0,-1]}
        #print(self.actioncoord[0][0])
        self.actioncoord = [[1,0],[-1,0],[0,1],[0,-1]]
        self._seed()
        self.viewer = None
        self.state = None
        self.state = list(np.random.randint(low,high,size=(2,)))
        #print(self.state)
        
        
    def _seed(self, seed = None):
        self.np_random,seed = seeding.np_random(int(time.time()))
        return [seed]
    
    def step(self,action):
        #assert self.action_space.contains(action),"%r (%s) invalid"%(action,type(action))
        #global reward, Tset, low, high, Twi, Tai, fa
        global A,A_,B, B_, low, high,rangelist
        action =  int(action)
        #print(action)
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
        #print(self.state)
        #self.steps_beyond_done = 0
        self._seed()
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
     
    