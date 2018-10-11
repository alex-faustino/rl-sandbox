# -*- coding: utf-8 -*-

"""
Abdul Alkurdi
v2
This is a custom Gridworld created by Abdul Alkurdi for HW1 of AE598 RL class.
This is a problem from Sutton and Barto RL book. I did some modifications
 id='AAgridworld-v0'
"""
import gym
from gym import core, spaces  
import numpy as np

import random




class AAgridworldEnv(core.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50}

    def __init__(self):
        
        
        #since it's a 5x5 grid and x&y are xy coordinates of each block
        self.x_max = 4
        self.y_max = 4
        
        
        # This sets the edges limit of each coordinate
        self.high = np.array(self.x_max)
        self.low = np.array(0)
                
        # 0 right 1 left
        # 2 up    3 down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5,5])
        
        
    def step(self, action, hard):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        observation_space = self.observation_space
        x, y = observation_space
        self.observation_space = (x,y)
        
        if hard and random.random() > 0.9:
            #print('action was replaced from ', action)
            #action=2 #
            action=random.randint(0,3)
            #print('to , \n'action)
        else:
            #print('nothing happened')
            pass
            
        if x==1 and y==4 :
            reward=10
            self.observation_space=[1,0]
        elif x==3 and y==4:
            reward=5
            self.observation_space=[3,2]
        elif (x==0 and action==1) or (x==4 and action==0) or (y==0 and action==3) or (y==4 and action==2):
            reward=-1
            self.observation_space=[x,y]
        else:
            reward=0
            if action ==0:
                x=x+1
            if action ==1:
                x=x-1
            if action ==2:
                y=y+1
            if action ==3:
                y=y-1
            self.observation_space=[x,y]
        done= False 
        return np.array(self.observation_space), reward, done, {}

    def reset(self):
        self.observation_space = [random.randint(1, self.high),random.randint(1, self.high)]
        self.steps_beyond_done = None
        self.num_steps=0
        return (self.observation_space)

    def render(self, mode='human'):
        
        output=[' o','a','o','b','o','\n','o','o','o','o','o','\n','o','o','o','o','o','\n','o','o','o','o','o','\n','o','o','o','o','o']

        [a,b]=self.observation_space
        text_position=(6*(4-b)+a)
        output[text_position]='x'
        print(*output)
        