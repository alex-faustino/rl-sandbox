from gym import core, spaces
from gym.utils import seeding, colorize
import numpy as np
import sys



class GOLEnv(core.Env):

    def __init__(self,nx=10,ny=10):
        self.nx = nx 
        self.ny = ny 
        self.nS = nx*ny
        
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nS//4) #alive or died
        #self.grid = numpy.zeros(self.nS, dtype='i').reshape(nx,ny)
        


    def reset(self,a):
        alive = np.random.randint(self.nS, size=a)
        self.s = np.zeros(self.nS,dtype='i')
        for i in alive:
            self.s[i] = 1
        return self.s

    def _to_s(self,i, j):
        return j*self.nx + i

    def _aliveNeighbours(self, i, j):
        """ Count the number of live neighbours around point (i, j). """
        count = 0 #
        for x in [i-1, i, i+1]:
            for y in [j-1, j, j+1]:
                if(x == i and y == j):
                    continue 
                if(x != self.nx and y != self.ny):
                    count += self.s[self._to_s(x,y)]
                    
                elif(x == self.nx and y != self.ny):
                    count += self.s[self._to_s(0,y)]
                elif(x != self.nx and y == self.ny):
                    count += self.s[self._to_s(x,0)]
                else:
                    count += self.s[self._to_s(0,0)]

        return count

    def step(self,a):
        sold = self.s.copy()
        
        for i in range(self.nx):
            for j in range(self.ny):
                pos = self._to_s(i,j)
                live = self._aliveNeighbours(i, j)
                if(sold[pos] == 1 and live < 2):
                    self.s[pos] = 0 # Dead from starvation.
                elif(sold[pos] == 1 and (live == 2 or live == 3)):
                    self.s[pos] = 1 # Continue living.
                elif(sold[pos] == 1 and live > 3):
                    self.s[pos] = 0 # Dead from overcrowding.
                elif(sold[pos] == 0 and live == 3):
                    self.s[pos] = 1 # Alive from reproduction.
        r = self.s.sum()>0
        
        return (self.s, r, r==0, None)
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        grid = np.zeros(self.nS, dtype='c').reshape(self.nx,self.ny)
        grid = [[c.decode('utf-8') for c in line] for line in grid]
        #row, col = self.s // self.ncol, self.s % self.ncol
        #grid[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        for x in range(self.nx):
            for y in range(self.ny):
                pos = self._to_s(x,y)
                if self.s[pos]:
                    grid[x][y] = colorize(str(self.s[pos]), "green", highlight=True)
                else:
                    grid[x][y] = colorize(str(self.s[pos]), "red", highlight=True)
                
        
        outfile.write("\n".join(''.join(line) for line in grid)+"\n")

        if mode != 'human':
            return outfile
  
