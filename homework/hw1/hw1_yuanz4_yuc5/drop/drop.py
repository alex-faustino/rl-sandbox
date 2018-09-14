import numpy as np
from gym import spaces
import gym
from gym.utils import seeding

class DropEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self):
        self.viewer = None
        self.drop_range = 11
        self.seed()

    def new_drop(self, freq):
        location = -1
        if self.count % freq == 0:
            location = np.random.randint(self.drop_range)
        self.count += 1
        return location

    def _step(self, action):
        done = self.done
        for i in self.items:
            i += np.array([1, 0])
        if action != -1:
            self.items.append(np.array([0, action]))
        if len(self.items) == 0:
            return self.board, 0, done, {}
        if self.items[0][0] >= self.drop_range:
            self.items.pop(0)
        if len(self.items) == 0:
            return self.board, 0, done, {}
        distance = self.board - self.items[0][1]
        if distance > 0:
            self.board -= 1
        elif distance < 0:
            self.board += 1
        reward = 0
        if self.items[0][0] == self.drop_range-1:
            if distance == 0:
                reward = 1
            else:
                reward = -1
        return self.board, reward, done, {}

    def _reset(self):
        self.done = 0
        self.board = (self.drop_range-1)/2
        self.items = []
        self.count = 0
        return self.board

    def _render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        size = self.drop_range
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(0, size+1, 0, size+1)
        for x in range(size+1):
            self.viewer.draw_line((0.5+x, 0.5), (0.5+x, 0.5+size))
        self.viewer.draw_line((0.5, 0.5+size), (0.5+size, 0.5+size))
        l,r,t,b = -.5, +.5, .1, -.1
        board_transform = rendering.Transform(translation=(self.board+1,0.5))
        board = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
        board.add_attr(board_transform)
        board.set_color(0,.8, .8)
        for i in self.items:
            x, y = i
            transform = rendering.Transform(translation=(y+1, size-x))
            circ = self.viewer.draw_circle(.4)
            circ.set_color(.8, .8, 0)
            circ.add_attr(transform)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        self.np_random, self._seed = seeding.np_random(seed)
