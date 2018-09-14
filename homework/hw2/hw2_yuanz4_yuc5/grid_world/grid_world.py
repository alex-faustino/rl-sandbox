import numpy as np
import gym
from gym.utils import seeding
from gym.envs.classic_control import rendering

class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, mode='Q', level='easy'):
        self.alpha = 0.7
        self.gamma = 0.5
        self.epsilon = 0.8
        # mode = 'Q' or 'SARSA', level = 'easy' or 'hard'
        self.mode = mode
        self.level = level
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
        self.q = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE, 4))
        self.seed()

    def action(self):
        x, y = self.state
        if self.level == 'hard':
            if np.random.rand() < self.epsilon:
                return np.random.randint(4)
        return np.argmax(self.q[x][y])

    def _step(self, action):
        done = self.done
        alpha = self.alpha
        gamma = self.gamma
        act = self.ACTIONS[action]
        next_state = self.state + act
        x, y = self.state
        current_q = self.q[x][y][action]
        reward = 0
        if (next_state == self.A_POS).all():
            next_state = self.A_PRIME_POS
            reward = 10
        if (next_state == self.B_POS).all():
            next_state = self.B_PRIME_POS
            reward = 5
        x_, y_ = next_state
        if x_ < 0 or x_ >= self.WORLD_SIZE or y_ < 0 or y_ >= self.WORLD_SIZE:
            reward = -1
            self.q[x][y][action] = current_q + alpha*(reward-current_q)
        else:
            if (self.mode == 'Q') or (self.level == 'easy'):
                next_q = np.max(self.q[x_][y_])
                self.q[x][y][action] = current_q + alpha*(reward+gamma*next_q-current_q)
            else:
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                    next_q = self.q[x_][y_][next_action]
                else:
                    next_q = np.max(self.q[x_][y_])
            self.q[x][y][action] = current_q + alpha*(reward+gamma*next_q-current_q)
            self.state = next_state
        return self.state, reward, done, {}

    def _reset(self):
        self.done = 0
        self.state = np.array(np.random.randint(self.WORLD_SIZE, size=2))
        return self.state

    def _render(self, mode='human', close=False):
        size = self.WORLD_SIZE
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(0, size+1, 0, size+1)
        for x in range(size+1):
            self.viewer.draw_line((0.5+x, 0.5), (0.5+x, 0.5+size))
        for y in range(size+1):
            self.viewer.draw_line((0.5, 0.5+y), (0.5+size, 0.5+y))
        x, y = self.state
        transform = rendering.Transform(translation=(y+1, size-x))
        circ = self.viewer.draw_circle(.4)
        circ.set_color(.8, .8, 0)
        circ.add_attr(transform)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw(self):
        size = self.WORLD_SIZE
        vfunc = np.vectorize(lambda t: f'{t:.3f}')
        result = vfunc(self.q)
        for row in range(size):
            print('-'*(17*size+1))
            var = []
            for column in range(size):
                cell = result[row][column]
                var.extend([cell[0], cell[1], cell[2], cell[3]])
            for i in range(size):
                print('|' + var[4*i+1].center(16), end='')
            print('|')
            for i in range(size):
                print('|' + var[4*i].ljust(8) + var[4*i+2].rjust(8), end='')
            print('|')
            for i in range(size):
                print('|' + var[4*i+3].center(16), end='')
            print('|')
        print('-'*(17*size+1))

    def seed(self, seed=None):
        self.np_random, self._seed = seeding.np_random(seed)
