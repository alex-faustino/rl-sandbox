import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

ACTION = ('W','E','N','S')
ACTIONFN = ('WEST','EAST','NORTH','SOUTH')
GRID = {
    "G1": [
        "OAOBO",
        "OOOOO",
        "OOObO",
        "OOOOO",
        "OaOOO",
    ],
}

class WORLDGRIDENV(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        
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
        #ACTION = ('W','E','N','S')
        def inc(row, col, act):
            a = ACTION[act]
            if a=='W': # west
                col = max(col-1,0)
            elif a=='S': # south
                row = min(row+1,nrow-1)
            elif a=='E': # est
                col = min(col+1,ncol-1)
            elif a=='N': # north
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter==b'A':
                        #A prime Location = (4,1)
                        li.append((1.0, to_s(4,1), 10, False))
                    elif letter==b'B':
                        #A prime Location = (2,3)
                        li.append((1.0, to_s(2,3), -5, False))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = False
                        rew = -((newrow==row) or (newcol==col))
                        li.append((1.0, newstate, rew, done))

        super(WORLDGRIDENV, self).__init__(nS, nA, P, isd)

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
