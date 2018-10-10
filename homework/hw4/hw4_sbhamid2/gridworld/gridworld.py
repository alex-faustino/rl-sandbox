import math
import gym
from gym import spaces, logger, utils
from gym.utils import seeding
import numpy as np
import sys
import random
from six import StringIO, b
from gym.envs.classic_control import rendering

GRID = ["00000",
        "00000",
        "00000",
        "00000", 
        "00000"]

'''
Problem-1, Homework-1: 
In this we design a grid world with random actions (left/right/up/down) given as input and 
occasional rewards are obtained based on the current location of the pointer. 
The grid considered is 5x5
Reward =10 is obtained the pointer lands on [0,1] and reward=5 is obtained when pointer lands on [0,3]

'''

### Related to hard and easy 
LEVEL = 'hard' 
LEVEL_EPS = 0.05

class GridWorldEnvNew(gym.Env):

    def __init__(self):

        self.viewer = None
        
        #### Defining the world and the reward states
        self.world = 5
        self.a_init = [0, 1]
        self.b_init = [0, 3]
        self.desc = np.asarray(GRID,dtype='c') 
        self.a_prime = [4, 1]
        self.b_prime = [2, 3]

        ### Initializing the variables
        self.state = None
        self.info = {}
        self.x = None
        self.y = None

        # left, up, right, down
        self.agent_act = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.world**2)
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = list(self.np_random.uniform(low=0, high=5, size=(2,)).astype(int) )
        init_pos = self.state[0]*self.world + self.state[1]
        #[spaces.Discrete(self.world).sample(), spaces.Discrete(self.world).sample()]
        return self.state, init_pos
    
    def step(self, act_no):
        
        #### current action and state
        if LEVEL == 'hard':
            myprob = random.uniform(0, 1) 
            if myprob < LEVEL_EPS:
                act_no = self.action_space.sample()
        
        action = self.agent_act[act_no]
        cur_state = self.state
        cur_x, cur_y = cur_state         
        
        if cur_state == self.a_init:
            next_state = self.a_prime
            x, y = next_state
            reward = 10
        elif cur_state == self.b_init:
            next_state = self.b_prime
            x, y = next_state
            reward = 5
        else:   
            next_state = list(map(sum, zip(cur_state, action))) #(state + action).tolist()
            x, y = next_state
            reward = 0
            if x < 0 or x >= self.world or y < 0 or y >= self.world:
                reward = -1.0
                next_state = cur_state
                x, y = next_state                

        ### Updating the variables for the next instant
        done = bool(0)
        ### next states are stored in these variables and not the current ones.
        self.state = next_state
        self.x = x
        self.y = y
        self.info['pos'] = self.x*self.world + self.y
        return self.state, reward, done, self.info

    def render(self, mode='human'):
        row, col = self.x, self.y
        
        #### Printing text grid
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.desc
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = '1'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        #outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        ### Printing rendered grid
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-1, self.world, -1, self.world)

        #if s is None: return None
        for i in range(self.world+1):
            self.viewer.draw_line((i-0.5, -0.5), (i-0.5, self.world-0.5))

        for i in range(self.world+1):
            self.viewer.draw_line((-0.5, i-0.5), (self.world-0.5, i-0.5))

        for i in range(self.world):
            for j in range(self.world):
                #print('i: ', i, 'j: ', j, 'val: ', desc[i][j])
                if (desc[i][j] =='1'):
                    j1 = rendering.Transform(rotation=0, translation=(j,(self.world-1)-i))
                    l1 = self.viewer.draw_polygon([(-0.1,-0.1), (-0.1,0.1), (0.1,0.1), (0.1,-0.1)])
                    l1.set_color(1, 0, 0)
                    l1.add_attr(j1)
 
        jtransform = rendering.Transform(rotation=0, translation=(col,(self.world-1)-row))
        circ = self.viewer.draw_circle(0.1)
        circ.set_color(0, 0, 0)
        circ.add_attr(jtransform)

        if mode != 'human':
            return outfile
        else: 
            return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def random_action(self): 
        act_x = random.randint(0,2)-1 #gym.spaces.Discrete(2).sample()
        if act_x ==0:
            act_y = 2*random.randint(0,1)-1
        else: 
            act_y = 0
        action = [act_x, act_y]
        return action 
    
