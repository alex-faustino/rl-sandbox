# -*- coding: utf-8 -*-

"""
Abdul Alkurdi
v2
This is a custom Gridworld created by Abdul Alkurdi for HW1 of AE598 RL class.
This is a problem from Sutton and Barto RL book.
"""

import gym
import numpy as np
import random
import gym.spaces as spaces



class AAgridworldEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50}

    def __init__(self):
        
        
        
        #since it's a 5x5 grid and x&y are xy coordinates of each block
        self.x_max = 5
        self.y_max = 5
        
        
        # This sets the edges limit of each coordinate
        high = np.array([self.x_max,self.y_max])
        low = np.array([1,1])
                
        # 1 right 2 left
        # 3 up    4 down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high,shape=[5,5])

           
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, y = state
        
        '''#this line sets the difficulty between easy and hard'''
        difficulty='hard' 
        
        self.state = (x,y)
        if difficulty=='easy' and random.random() > 0.9:
            pass
        else:
            action=random.randint(1,4)
            
            
        if x==2 and y==5 :
            reward=10
            self.state=[2,1]
        elif x==4 and y==5:
            reward=5
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
        self.state = [random.randint(1, high[0]),random.randint(1, high[1])]
        self.steps_beyond_done = None
        self.num_steps=0
        return (self.state)

    def render(self, mode='human'):
        
        output=[' o','a','o','b','o','\n','o','o','o','o','o','\n','o','o','o','o','o','\n','o','o','o','o','o','\n','o','o','o','o','o']

        [a,b]=state
        text_position=(5*(5-b)+a)
        output[text_position]='x'
        print(*output)
        