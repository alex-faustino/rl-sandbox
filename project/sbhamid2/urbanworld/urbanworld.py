import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
import sys
import random
from six import StringIO, b
from gym.envs.classic_control import rendering

import pyglet
import toolutils
import simulations
import coordutils
import georinex as gr
import pyglet

'''
AE 598 RL FINAL PROJECT
Author: Sriramya Bhamidipati
Dated: 11/23/2018
Title: Autonomous Path Planning to reduce GPS Integrity Risk while navigating in Urban Environments
'''

### Inputs to be decided by the user at the start of the training process

# 1. Simulating the urban buildings and their corresponding heights
WORLD = [['0']*7]*7 ### Size of the grid world
BLK_LEN = 100.0 ### Length of the building block, i.e., from one intersection to another
WIDTH = 40 ### width/length of the buildings            

# 2. Reference LLA point for the ENU frame from ECEF frame
Ref_LLA = [[40.1064102],[-88.2265052],[16.25]] ### Reference point for conversion from ECEF to ENU frame
Ref_ECEF = coordutils.LLA_to_ECEF(Ref_LLA[0], Ref_LLA[1], Ref_LLA[2]) ### The Ref point in ECEF frame required later on 

# 3.Load the ephemeris file from the data folder -> obtained from CDDIS
### In this work static satellites are considered
t_gps = 172820.0 ### Time at which the static GPS satellite positions are calculated
EL_MASK = 5.0 ### In deg and not radians. elevation angles below which satellites are not considered as visible 
p_rinexfn = 'data/brdc3530.17n' ### the rinex files from which satellite orbital parameters are calculated
EPHEM = gr.load(p_rinexfn) ### Using georinex library load the rinex files

# 4. Parameters regarding the robot
SPEED = 10 ### Speed of the robot 
ALT_RX = 4 ### Constant height of the robot is considered, i.e., 2D motion 
GOAL = np.array([BLK_LEN*1, BLK_LEN*(len(WORLD)-1), ALT_RX]) ### Fixed goal is considered for reaching a point

class UrbanWorldEnv(gym.Env):

    def __init__(self):

        self.viewer = None

        ### Action and observation space, discrete: left, up, right, down, respectively
        self.world = len(WORLD) + 1
        self.HEIGHT = None
        
        self.agent_act = [np.array([0, -1, 0]), \
                          np.array([-1, 0, 0]), \
                          np.array([0, 1, 0]), \
                          np.array([1, 0, 0])]
        self.action_space = spaces.Discrete(4) ### 4 actions
        self.observation_space = spaces.Discrete( self.world**2) ### 25 states
        
        ### Initializing the variables
        self.plot = None ### whether or not 3D position plot of buildings, satellites and receiver is plotted
        ### if "obs": [relative pos, pseudoranges and distances] are considered as observations obtained from the environment; else: observations from step function are 3D positions, 
        self.mode = None
        ### Indicates whether its easy environment-> deterministic
        ### or hard-> some probability of moving in a direction different to the desired action
        self.version = None
        ### 3D position of the robot-> true
        self.state = None
        ### Index of the discrete states starting from 0 -> len(world)^2
        self.pos = None
        ### Observations provided to the robot, i.e., sensor data
        self.observation = None
        self.observation_dim = None
        ### Variable that stores the previous distance to the target
        self.dist_to_target = None
        ### Other information
        self.info = {}
        
        #### Defining the world and the reward states
        self.hard_prob = 0.1
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = list(self.np_random.uniform(low=1, high=len(WORLD)-1, size=(2,)).astype(int) ) + [ALT_RX/BLK_LEN]
        self.pos = self.state[0]*self.world + self.state[1]    ### This is still in 2-D... will think about this later 
        self.dist_to_target = np.linalg.norm( [BLK_LEN*self.state[0], BLK_LEN*self.state[1], BLK_LEN*self.state[2]] - GOAL)
        print('Reset is called: ', self.state)
        return self.state, self.pos
    
    def step(self, act_no):
        
        #### current action and state
        if self.version =='hard':
            hard_val = random.uniform(0, 1)
            if hard_val <self.hard_prob:
                #print('Random action is taken by robot')
                act_no = self.action_space.sample()
        
        ### The action chose and the current state
        action = self.agent_act[act_no]  
        cur_state = self.state

        next_state = list(map(sum, zip(cur_state, action))) 
        x, y, z = next_state
        reward = 0
        if x < 1 or x >= (self.world-1) or y < 1 or y >= (self.world-1):
            next_state = cur_state
            x, y, z = next_state
            
        ### states of the system after action is taken are denoted by 
        self.state = next_state
        self.pos = self.state[0]*self.world + self.state[1]
        self.info['pos'] = self.pos

        #### Plotting the satellites visible in the sky
        visible_sats = {}
        Rx_ENU = BLK_LEN*np.mat([[next_state[0]], [next_state[1]], [next_state[2]]])
        self.info['Rx_ENU'] = [Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0]]

        #print('cur_state: ', cur_state, 'action: ', action, 'next_state', self.state)
        Rx_ECEF = coordutils.ENU_to_ECEF(Ref_ECEF, Rx_ENU)
        visible_sats = simulations.gen_satpositions(EPHEM, t_gps, EL_MASK, Ref_ECEF, Rx_ECEF)
            
        pseudoranges, heights = simulations.compute_multipath(Rx_ENU, abs(action), BLK_LEN, WIDTH, \
                                                     self.HEIGHT, visible_sats, self.pos, plot=self.plot)
        
        ### Trilateration to estimate the GPS position from pseudoranges
        Rx_est = simulations.perform_least_sqrs( visible_sats['satLoc'], pseudoranges, np.zeros(3) )
        
        ### EKF measurement update to estimate the GPS position from pseudoranges
        ### EKF libraries are not uploaded to GITHUB to run this
        #Rx_est = ekf.measurement_update( visible_sats['satLoc'], pseudoranges, np.zeros(3) )
        self.info['Rx_est'] = Rx_est
        self.info['rel_pos'] = [Rx_est[0]-GOAL[0], Rx_est[1]-GOAL[1], Rx_est[2]-GOAL[2]]
        self.observation = self.info['rel_pos'] +list(pseudoranges.values()) + list(heights.values())
       
        ### Integrity files are related to research group files 
        ### 1. Reward attained from the GPS integrity risk metric
        ### The corresponding RAIM libraries are not uploaded to github
        #reward_raim = raim.solution_sep(visible_sats['satLoc'], pseudoranges)
        
        ### Initially tried out direct difference between estimated and true position as input
        reward_raim = -0.1*np.linalg.norm(Rx_est- self.info['Rx_ENU']) ### Or integrity when I add that part
            
        
        ### Check how far from the target
        #print('curr: ', cur_state, 'next: ', next_state, 'GOAL: ', GOAL/100.0, \
        #      'reward: ', self.dist_to_target, np.linalg.norm(self.info['Rx_ENU']- GOAL), \
        #      'raim: ', reward_raim)
        
        #print('reward_dist: ', reward_dist, ', reward_goal: ', reward_goal, \
        #      'old dist_to_target: ', self.dist_to_target, 'curr dist_to_target: ', np.linalg.norm(self.info['Rx_ENU']- GOAL))
        
        ### 2. Difference in distance between the previous position to goal and current position to goal 
        ### c_r = 0.2 
        reward_dist =  0.2*(self.dist_to_target-np.linalg.norm(self.info['Rx_ENU']- GOAL))        
        self.dist_to_target = np.linalg.norm(self.info['Rx_ENU']- GOAL)
                       
        ### 3. Reward goal is attained when the goal is reached -> in this case (800, 800, 4)
        if ( abs(self.dist_to_target) <1e-3):
            done = 1
            reward_goal = 50
        else: 
            done = 0
            reward_goal = 0
            
        ### Summation of above three components gives the total reward
        reward = reward_dist + reward_goal + reward_raim 
        #print('reward: ', reward, 'r_dist: ', reward_dist, 'r_goal: ', reward_goal)
        #print('Bef state', cur_state, 'Action: ', action, 'After state: ', self.state, 'GOAL: ', GOAL, 'done: ', done)
                       
        if (self.mode=='obs'): 
            self.observation_dim = len(self.observation)
            ### In this example measurement from the system is same as state therefore, self.state is obtained as output
            return self.observation, reward, done, self.info
        else: 
            self.observation_dim = len(self.state)
            ### The observation/input to RL algorithm are the 3D position inputs of the robot
            return self.state, reward, done, self.info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def copy(self):
        c = UrbanWorldEnv()
        c.state = self.state.copy()
        c.HEIGHT = self.HEIGHT.copy()
        
        return c            
            
    def render(self, mode='human'):
        row, col, _ = self.state
        
        ### Printing rendered grid
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-0.5, self.world-0.5, -0.5, self.world-0.5)
        
        for i in range(1,self.world-1):
            self.viewer.draw_line((i, 1), (i, self.world-2))

        for i in range(1,self.world-1):
            self.viewer.draw_line((1, i), (self.world-2, i))

        for i in range(self.world-1):
            for j in range(self.world-1):
                j1 = rendering.Transform(rotation=0, translation=(j+0.5,(self.world-1.5)-i))
                l1 = self.viewer.draw_polygon([(-0.3,-0.3), (-0.3,0.3), (0.3,0.3), (0.3,-0.3)])
                l1.set_color(1-(self.HEIGHT[i][j]/100), 1-(self.HEIGHT[i][j]/100), 1-(self.HEIGHT[i][j]/100))
                l1.add_attr(j1)
 
        jtransform = rendering.Transform(rotation=0, translation=(col,(self.world-1)-row))
        circ = self.viewer.draw_circle(0.1)
        if (self.pos== (GOAL[0]*self.world + GOAL[1])/BLK_LEN ):
            circ.set_color(0, 1, 0)
        else: 
            circ.set_color(0, 0, 0)
        circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')