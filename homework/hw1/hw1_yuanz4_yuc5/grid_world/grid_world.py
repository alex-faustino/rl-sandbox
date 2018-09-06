import numpy as np
from gym import spaces
import gym
from gym.utils import seeding

class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self):
        self.viewer = None
        # left, up, right, down
        self.ACTIONS = [np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1]),
                        np.array([1, 0])]
        self.WORLD_SIZE = 5
        self.PROB = 10
        self.A_POS = np.array([0, 1])
        self.A_PRIME_POS = np.array([4, 1])
        self.B_POS = np.array([0, 3])
        self.B_PRIME_POS = np.array([2, 3])
        self.seed()

    def random(self, tf):
        self.random = tf

    def action(self):
        if self.last_action == None:
            self.last_action = 0
            return self.last_action
        if self.random == True:
            prob = np.random.randint(self.PROB)
            if prob == 0:
                self.last_action = np.random.randint(4)
                return self.last_action
        if (self.old_state == self.new_state).all():
            action = (self.last_action+1) % 4
            self.last_action = action
        return self.last_action

    def _step(self, action):
        self.old_state = self.new_state
        done = self.done
        if (self.old_state == self.A_POS).all():
            self.new_state = self.A_PRIME_POS
            return self.A_PRIME_POS, 10, done, {}
        if (self.old_state == self.B_POS).all():
            self.new_state = self.B_PRIME_POS
            return self.B_PRIME_POS, 5, done, {}
        act = self.ACTIONS[action]
        next_state = self.old_state + act
        x, y = next_state
        if x < 0 or x >= self.WORLD_SIZE or y < 0 or y >= self.WORLD_SIZE:
            reward = -1.0
            next_state = self.old_state
        else:
            reward = 0
        self.new_state = next_state
        return next_state, reward, done, {}

    def _reset(self):
        self.done = 0
        self.last_action = None
        self.new_state = np.array(np.random.randint(5, size=2))
        self.old_state = None
        return self.new_state

    def _render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        size = self.WORLD_SIZE
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(0, size+1, 0, size+1)
        for x in range(size+1):
            self.viewer.draw_line((0.5+x, 0.5), (0.5+x, 0.5+size))
        for y in range(size+1):
            self.viewer.draw_line((0.5, 0.5+y), (0.5+size, 0.5+y))
        x, y = self.new_state
        transform = rendering.Transform(translation=(y+1, size-x))
        circ = self.viewer.draw_circle(.4)
        circ.set_color(.8, .8, 0)
        circ.add_attr(transform)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        self.np_random, self._seed = seeding.np_random(seed)
