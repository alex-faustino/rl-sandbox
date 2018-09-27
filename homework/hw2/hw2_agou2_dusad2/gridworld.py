from gym import core, spaces
from gym.utils import seeding
import numpy as np


class GridWorld(core.Env):

    """
    **STATE:**
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    **REFERENCE:**
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    def __init__(self, grid_size=[5,5], is_hard=False):
        
        self.is_hard = is_hard
        # state space
        self.observation_space = spaces.Box(low=np.array([1,1]), high=np.array(grid_size), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.position = self.observation_space.sample()
        
        self.a = np.array([1,2])
        self.a_prime = np.array([5,2])
        
        self.b = np.array([1,4])
        self.b_prime = np.array([3,4])
        
        self._action_delta = {
            0: np.array([0, 1]),
            1: np.array([0,-1]),
            2: np.array([1,0]),
            3: np.array([-1,0])
        }
        id
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = self.observation_space.sample()
        return self.position

    def step(self, a):
        # next_position, reward 
        if np.array_equal(self.position,self.a):
            self.position = self.a_prime
            return self.position, 10, False, None
        elif np.array_equal(self.position, self.b):
            self.position = self.b_prime
            return self.position, 5, False, None
        
        if self.is_hard:
            is_random = np.random.choice([True, False], p=[0.1, 0.9])
            if is_random:
                a = self.action_space.sample()
        next_position = self.position + self._action_delta[a]
        if not self._is_in_bound(next_position):
            return self.position, -1, False, None
        self.position = next_position
        return self.position, 0, False, None
        
    def _is_in_bound(self, position):
        return self.observation_space.contains(position)

    def render(self, mode='ansi'):
        return self.print_grid()

    def close(self):
        if self.viewer: self.viewer.close()
    
    def print_grid(self):
        empty_space = "| "
        current_space = "|o"
        
        grid = ""
        row = ""
        for x in range(self.observation_space.low[0], self.observation_space.high[0]+1):
            for y in range(self.observation_space.low[1],self.observation_space.high[1] +1):
                if np.array_equal(np.array([x,y]), self.position):
                    grid += current_space
                else:
                    grid += empty_space
            grid += "|\n" + row
        return grid

    def show_trajectory(self, states):
        states = [str(state) for state in states]
        prev_states = states[:-1]
        last_state = states[-1]
        empty_space = "| "
        prev_space = "|."
        curr_space = "|o"
        
        grid = ""
        row = ""
        for x in range(self.observation_space.low[0], self.observation_space.high[0]+1):
            for y in range(self.observation_space.low[1],self.observation_space.high[1] +1):
                if str(np.array([x,y])) == last_state:
                    grid += curr_space
                elif str(np.array([x,y])) in prev_states:
                    grid += prev_space
                else:
                    grid += empty_space
            grid += "|\n" + row
        return grid