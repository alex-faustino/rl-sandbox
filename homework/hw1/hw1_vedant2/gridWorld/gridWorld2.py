# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 05:40:09 2018

@author: Vedant
"""

import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

GRID = {
    "G1": [
        "EAEBE",
        "ESSSE",
        "ESSSE",
        "ESSSE",
        "EEEEE",
    ],
}

class gridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,hard = False):
        
        self.desc = desc = np.asarray(GRID['G1'],dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1, 0, 5, 10)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc != b'1').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if hard:
                if np.random.rand(1,1)>0.8:
                    a = np.random.randint(0,4)
            probability = 1.0;
            reward = 0;
            #check if in A or B
            if (row == 0 & col == 1 ):
                reward = 10;
                row = nrow-1;
                probability = 1.0;
            elif (row == 0 & col == 3 ):
                reward = 5;
                row = 2;
                probability = 1.0;
            else:
                if a==0: # West
                    col = col-1;
                    if (col<0):
                        col = 0;
                        reward = -1;
                elif a==1: # South
                    row = row+1
                    if (row>nrow-1):
                        row = nrow-1;
                        reward = -1;
                elif a==2: # East
                    col = col+1;
                    if(col>ncol-1):
                        col = ncol-1;
                        reward = -1;
                elif a==3: # North
                    row = row-1;
                    if (row<0):
                        row = 0;
                        reward = -1;
            return (row, col,reward, probability)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    newrow, newcol, rew, probability = inc(row, col, a)
                    newstate = to_s(newrow, newcol)
                    li.append((probability, newstate, rew, False))

        super(gridWorld, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        row, col = self.s // self.ncol, self.s % self.ncol
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        for col in range(self.ncol):
            for row in range(self.nrow):
                desc[row][col] = utils.colorize(desc[row][col], "cyan", highlight=True)
                
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(ACTIONFN[self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
