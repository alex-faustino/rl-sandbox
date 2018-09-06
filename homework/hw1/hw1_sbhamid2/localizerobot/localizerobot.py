"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger, utils
from gym.utils import seeding
import numpy as np
import sys
from six import StringIO, b
from scipy.optimize import least_squares
 
## 10x10 grid input by the user indicating the map in which they want to localize the robot
## same should be mentioned in the self.world input
GRID = ["xxxxxxxxxx",
        "x00000000x",
        "x00000000x",
        "x00000000x", 
        "x000000xtx",
        "x00000000x",
        "xt0000000x", 
        "x00000000x",
        "x000t0000x",
        "xxxxxxxxxx"]
'''
In this example, I develop a wireless localization framework, where the signals transmitted by the 
beacons (indicated by "t" in the above grid) are received by the robot. In the ipynotebook, the blue color circles in the 
env.render() indicate the transmitter locations whereas red rectangles are the walls/obstacles of some sort. 
Both the robot and 3 transmitters are placed in the 2d grid. The boundary walls of the grid are indicated by "x". 
The robot processed the information from the beacons to estimate its current position via least squares. 
The aim of this is to reach the target position. 
However, one of the transmitter is prone to multipath because of which the state position estimated by the robot has errors. 
Random actions are given to the robot such that the robot either steps up/down/left/right. 
The reward function is calculated based on the distance between its true location from the target location.
In this code, state=true location of robot, pos = estimated location of the robot via trilateration using transmitter signals 
'''
class LocalizeEnvNew(gym.Env):

    def __init__(self):

        self.viewer = None
        self.world = 10
        self.desc = np.asarray(GRID,dtype='c') 
        self.discount = 1.0

        self.yp_refl = [1.5, 8.5]
        self.yn_refl = [6.5, 8.5]
        
        # left, up, right, down
        self.agent_act = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
        self.prob = 0.25

        self.state = None
        self.x = None
        self.y = None
        self.pos = None
        self.target = [8,8]

        #self.action_space = np.array([spaces.Discrete(2).sample(), spaces.Discrete(2).sample()])
        self.tx = []
        for i in range(self.world):
            for j in range(self.world): 
                #i is row and j is column of the grid
                if (self.desc[i][j].decode('UTF-8') =='t'):
                    self.tx.append([i,j])
        print('TXs: ', self.tx)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = list(self.np_random.uniform(low=0, high=5, size=(2,)).astype(int) )
        self.pos = self.state
        self.dist = np.linalg.norm(np.array(self.state)-np.array(self.target)) 
        #[spaces.Discrete(self.world).sample(), spaces.Discrete(self.world).sample()]
        return self.state

    def sys_eqs(self, p):
        eq_arr = []
        for i in range(len(self.tx)):
            eq_arr.append(np.sqrt((p[0]-self.tx[i][0])**2+(p[1]-self.tx[i][1])**2)-self.meas[i])
        return np.array(eq_arr)
        #return np.array( [np.sqrt((p[0]-self.tx[0][0])**2+(p[1]-self.tx[0][1])**2)-self.meas[0], 
        #    np.sqrt((p[0]-self.tx[1][0])**2+(p[1]-self.tx[1][1])**2)-self.meas[1], 
        #    np.sqrt((p[0]-self.tx[2][0])**2+(p[1]-self.tx[2][1])**2)-self.meas[2] ] )

    def step(self, action):
        state = self.state
        
        next_state = list(map(sum, zip(state, action))) #(state + action).tolist()
        x, y = next_state
        
        if (self.desc[x][y].decode('UTF-8') =='x') or (self.desc[x][y].decode('UTF-8') =='t'):
            next_state = state
            x, y = next_state

        self.meas = (np.linalg.norm(np.array(self.tx)-np.array(next_state), axis=1))

        #### Introducing multipath in the system
        #print('state: ', x, y)
        if(x<self.world/2):
            #print('pos_axis')
            self.meas[0] = np.linalg.norm(np.array(self.tx[0])-np.array(self.yp_refl)) + np.linalg.norm(np.array(self.yp_refl)-np.array(next_state))
        else: 
            #print('neg_axis')
            self.meas[0] = np.linalg.norm(np.array(self.tx[0])-np.array(self.yn_refl)) + np.linalg.norm(np.array(self.yn_refl)-np.array(next_state))
        
        #Here, pos indicates the estimated position calculated via least squares
        est_pos =  least_squares(self.sys_eqs, np.array(self.pos))
        self.pos = [est_pos.x[0], est_pos.x[1]]
        print('true: ', next_state, 'est: ', self.pos, 'cost: ', est_pos.cost)    
        
        #Reward is assigned based on how close the robot is to its target location
        # If it moved closer, positive reward proportional to the distance is assigned
        # If it moved farther, negative reward is assigned
        cur_dist =  np.linalg.norm(np.array(self.state)-np.array(self.target)) 
        reward = self.discount*(self.dist - cur_dist)      
        
        #### Update the variables with the current statistics
        # Here, state indicates the true position of the robot in the 2d grid
        self.state = next_state
        self.x = x
        self.y = y
        self.dist = cur_dist

        ## if the robot reaches the target location
        if (self.dist <1e-4): 
            done = bool(1)
        else: 
            done = bool(0)
        
        return self.meas, reward, done, list(np.array(self.state)-np.array(self.pos))

    def render(self, mode='human'):

        from gym.envs.classic_control import rendering

        s = self.pos

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-1, self.world+1, -1, self.world+1)

        if s is None: return None
        for i in range(self.world+1):
            self.viewer.draw_line((i-0.5, -0.5), (i-0.5, self.world-0.5))

        for i in range(self.world+1):
            self.viewer.draw_line((-0.5, i-0.5), (self.world-0.5, i-0.5))

        for i in range(self.world):
            for j in range(self.world):
                if (self.desc[i][j].decode('UTF-8') =='x'):
                    #print('i: ', i, 'j: ', j, 'val: ', self.desc[i][j])
                    j1 = rendering.Transform(rotation=0, translation=(j,(self.world-1)-i))
                    l1 = self.viewer.draw_polygon([(-0.1,-0.1), (-0.1,0.1), (0.1,0.1), (0.1,-0.1)])
                    l1.set_color(1, 0, 0)
                    l1.add_attr(j1)

                if (self.desc[i][j].decode('UTF-8') =='t'):
                    j2 = rendering.Transform(rotation=0, translation=(j,(self.world-1)-i))
                    l2 = self.viewer.draw_circle(0.25)
                    l2.set_color(0, 0.8, 0.8)
                    l2.add_attr(j2)

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        row, col = self.x, self.y 
 
        jtransform = rendering.Transform(rotation=0, translation=(col,(self.world-1)-row))
        circ = self.viewer.draw_circle(0.1)
        circ.set_color(0, 0, 0)
        circ.add_attr(jtransform)

        desc = self.desc 
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = '1'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
