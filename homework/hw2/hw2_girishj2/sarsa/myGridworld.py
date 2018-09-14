'''
#--------------------------------------------------------------------
Author : Girish Joshi
Email: girishj2@illinois.edu

This code Implements Grid-World Enviroment from Example 3.5 
from book "Reinforcement Learning Second Edition" Rich Sutton and Barto
#--------------------------------------------------------------------
'''
import numpy as np
import random 
import sys
 
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class gridWorldEnv():

    def __init__(self, mode):
        self.shape = [5,5]

        self.nCell = np.prod(self.shape)
        self.nAction = 4

        self.max_X = self.shape[0]
        self.max_Y = self.shape[1]

        self.grid = np.arange(self.nCell).reshape(self.shape)
        it = np.nditer(self.grid, flags=['multi_index'])
        self.P = {}

        self.state = 0

        self.mode = mode

        while not it.finished:
            
            is_goal = False
            is_wall = False

            sIdx = it.iterindex
            self.y, self.x = it.multi_index

            self.P[sIdx] = {a:[] for a in range(self.nAction)}

            is_goal = lambda s: sIdx == 1 or sIdx == 3
            is_wall = lambda x,y: (y == 0 and UP) or (x == 0 and LEFT) or (y == (self.max_Y - 1) and DOWN) or (x == (self.max_X - 1) and RIGHT)
           
            if is_goal(sIdx):
                if sIdx == 1:
                    reward = 10
                elif sIdx == 3:
                    reward = 5
            elif is_wall(self.x, self.y):
                reward = -1
            else:
                reward = 0

            # High Reward State
            if is_goal(sIdx):
                if sIdx == 1:
                    self.P[sIdx][UP] = [(1.0, 21, reward, True)]
                    self.P[sIdx][RIGHT] = [(1.0, 21, reward, True)]
                    self.P[sIdx][DOWN] = [(1.0, 21, reward, True)]
                    self.P[sIdx][LEFT] = [(1.0, 21, reward, True)]
                elif sIdx == 3:
                    self.P[sIdx][UP] = [(1.0, 13, reward, True)]
                    self.P[sIdx][RIGHT] = [(1.0, 13, reward, True)]
                    self.P[sIdx][DOWN] = [(1.0, 13, reward, True)]
                    self.P[sIdx][LEFT] = [(1.0, 13, reward, True)]

            # Not a High Reward State
            else:
                ns_up = sIdx if self.y == 0 else sIdx - self.max_X
                ns_right = sIdx if self.x == (self.max_X - 1) else sIdx + 1
                ns_down = sIdx if self.y == (self.max_Y - 1) else sIdx + self.max_X
                ns_left = sIdx if self.x == 0 else sIdx - 1
                self.P[sIdx][UP] = [(1.0, ns_up, reward, is_goal(ns_up))]
                self.P[sIdx][RIGHT] = [(1.0, ns_right, reward, is_goal(ns_right))]
                self.P[sIdx][DOWN] = [(1.0, ns_down, reward, is_goal(ns_down))]
                self.P[sIdx][LEFT] = [(1.0, ns_left, reward, is_goal(ns_left))]

            it.iternext()

    def step(self,u):
        u = self.action_selection(u)
        info = self.P[self.state][u]
        self.state = info[0][1]
        reward = info[0][2]
        done = info[0][3]
        return self.state, reward, done

    def render(self):
        it = np.nditer(self.grid, flags=['multi_index'])
        outfile = sys.stdout
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.state == s:
                output= 'x'
            elif s == 1 or s == 3:
                output = 'T'
            else:
                output = 'o'

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

    def action_space_sample(self):
        return np.random.randint(0,4)

    def action_selection(self,action):
        if self.mode == True:
            action = action
        else:
            if np.random.uniform(0,1,size=None) < 0.5:
                action = action
            else:
                action = np.random.randint(0,4)
        return action

    def action_space_size(self):
        return 4

    def observation_space_size(self):
        return self.nCell

    def reset(self):
        self.state = np.random.randint(low=0, high=self.nCell)
        return self.state