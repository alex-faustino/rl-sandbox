# -*- coding: utf-8 -*-
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

MAP = {["EAEBE",
        "ESSSE",
        "ESSSE",
        "ESSSE"]}

class GirdWorldEnv(discrete.DiscreteEnv):
    """
    The cells of the grid correspond to the states of the environment. At
each cell, four actions are possible: north, south, east, and west, which deterministically
cause the agent to move one cell in the respective direction on the grid. Actions that
would take the agent o↵ the grid leave its location unchanged, but also result in a reward
of −1. Other actions result in a reward of 0, except those that move the agent out of the
special states A and B. From state A, all four actions yield a reward of +10 and take the
agent to A0. From state B, all actions yield a reward of +5 and take the agent to B0.
    """

    def __init__(self):
        self.nrow, self.ncol = 5, 5

        nA = 4
        nS = nrow * ncol

        #isd = np.array(desc == b'S').astype('float64').ravel()
        #isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a):
            if a==0: # West
                col = max(col-1,0)
            elif a==1: # South
                row = min(row+1,nrow-1)
            elif a==2: # East
                col = min(col+1,ncol-1)
            elif a==3: # North
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
						newrow, newcol = inc(row, col, a)
						newstate = to_s(newrow, newcol)
						newletter = desc[newrow, newcol]
						done = bytes(newletter) in b'GH'
						rew = float(newletter == b'G')
						li.append((1.0, newstate, rew, done))
        sNorther(GirdWorldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["West","South","East","North"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile