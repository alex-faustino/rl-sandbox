import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class GridWorldHardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # set action space
        # 0 = North
        # 1 = East
        # 2 = South
        # 3 = West
        self.action_space = spaces.Discrete(4)

        # set observation space
        self.obs_high = 4
        self.observation_space = spaces.Box(low=0, high=self.obs_high,
                                            shape=[self.obs_high, self.obs_high], dtype=np.uint8)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # randomly choose another action 10% of the time
        if np.random.rand() < .1:
            action = self.action_space.sample()

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x1, x2 = self.state

        # handle special cases
        if x1 == 1 and x2 == 4:
            reward = 10
            self.state = [1, 0]
        elif x1 == 3 and x2 == 4:
            reward = 5
            self.state = [3, 2]
        # handle edge cases
        elif (x1 == 0 and action == 3) or (x1 == self.obs_high and action == 1) \
                or (x2 == 0 and action == 2) or (x2 == self.obs_high and action == 0):
            reward = -1
            self.state = [x1, x2]
        # normal cases
        else:
            reward = 0
            if action == 0:
                x2 += 1
            if action == 1:
                x1 += 1
            if action == 2:
                x2 += -1
            if action == 3:
                x1 += -1
            self.state = [x1, x2]

        done = False
        return np.array(self.state), reward, done, []

    def reset(self):
        self.state = self.np_random.randint(low=0, high=self.obs_high, size=(2,))
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.obs_high + 1
        scale = screen_width / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # draw grid
            for i in range(self.obs_high + 2):
                v_line = self.viewer.draw_line((i * scale, 0.), (i * scale, screen_height))
                h_line = self.viewer.draw_line((0., i * scale), (screen_width, i * scale))
                self.viewer.add_geom(v_line)
                self.viewer.add_geom(h_line)
            # draw the agent
            l, r, t, b = -scale / 2 + .5, scale / 2 - .5, scale / 2 - .5, -scale / 2 + .5
            agent = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.agent_trans = rendering.Transform()
            agent.add_attr(self.agent_trans)
            agent.set_color(.5, .5, .8)
            self.viewer.add_geom(agent)

        if self.state is None:
            return None

        self.agent_trans.set_translation(self.state[0] * scale + scale / 2, self.state[1] * scale + scale / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
