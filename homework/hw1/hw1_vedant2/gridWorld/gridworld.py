
"""
Created on Thu Aug 30 11:07:47 2018

@author: Vedant
"""

import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

West = 0
South = 1
East = 2
North = 3

MAP = ["EAEBE",
        "ESSSE",
        "ESSSE",
        "ESSSE"]

class gridWorld(discrete.DiscreteEnv):
    """
    The cells of the grid correspond to the states of the environment. At
each cell, four actions are possible: north, south, east, and west, which deterministically
cause the agent to move one cell in the respective direction on the grid. Actions that
would take the agent o↵ the grid leave its location unchanged, but also result in a reward
of −1. Other actions result in a reward of 0, except those that move the agent out of the
special states A and B. From state A, all four actions yield a reward of +10 and take the
agent to A0. From state B, all actions yield a reward of +5 and take the agent to B0.
    """
metadata = {'render.modes': ['human', 'ansi']}

def __init__(self):
    
    self.desc = desc = np.asarray(MAP,dtype='c');    
    self.nrow, self.ncol = nrow, ncol = desc.shape;
    self.reward_range = (-1, 0, 5, 10)
    
    nA = 4;
    nS = nrow * ncol;
    
    isd = np.array(desc == b'S').astype('float64').ravel();
    isd /= isd.sum();

    P = {s : {a : [] for a in range(nA)} for s in range(nS)}

    def to_s(row, col):
        return row*ncol + col;

    def inc(row, col, a):
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

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = self.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
    if self.lastaction is not None:
        outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
    else:
        outfile.write("\n")
    outfile.write("\n".join(''.join(line) for line in desc)+"\n")

    if mode != 'human':
        return outfile