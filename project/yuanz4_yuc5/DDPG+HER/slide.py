import numpy as np
import numpy.linalg as la
import math

class SlideEnv():
    def __init__(self):
        self.table = 100.
        self.ball_radius = 5.
        self.goal_radius = 5.
        self.probe_start = np.array([50., 50.])
        self.ball_start = 25.
        self.move_range = 10.
        self.slide_rate = 0.
        self.observation_space = 6
        self.action_space = 2

    def reset(self):
        obs = {}
        self.probe = np.copy(self.probe_start)
        obs['observation'] = self.probe
        self.ball = np.random.uniform(-self.ball_start, self.ball_start, 2)
        self.ball += self.probe_start
        obs['achieved_goal'] = self.ball
        self.goal = np.random.uniform(0, self.table, 2)
        obs['desired_goal'] = self.goal
        return np.block([self.probe, self.ball, self.goal])

    def step(self, action):
        reward = -1
        done = 0
        success = ''
        dxdy = self.move_range * action
        x0y0 = self.probe
        bxby = self.ball - x0y0
        projection = np.dot(bxby, dxdy)*dxdy/la.norm(dxdy)**2
        h = la.norm(bxby-projection)
        obs = {}
        self.probe += dxdy
        obs['observation'] = self.probe
        if h < self.ball_radius:
            dxdy_len = la.norm(dxdy)
            projection_len = la.norm(projection)
            cross_1 = math.sqrt(self.ball_radius**2-h**2)
            cross_2 = dxdy_len - projection_len
            cross = cross_1 + cross_2
            if cross > 0:
                self.ball += cross*dxdy/la.norm(dxdy)*(self.slide_rate+1)
            x = self.ball[0]
            y = self.ball[1]
            if x < 0 or x > self.table or y < 0 or y > self.table:
                reward = -1
                done = 1
            distance = la.norm(self.ball-self.goal)
            if distance < (self.ball_radius+self.goal_radius):
                success = 'congratulation!'
                reward = 0
                done = 1
        obs['achieved_goal'] = self.ball
        return np.block([self.probe, self.ball, self.goal]), reward, done, success

    def cal_reward(self, st0, st1):
        xbyb = st0[2:4]
        xgyg = st1[2:4]
        distance = la.norm(xbyb-xgyg)
        if distance < self.move_range:
            return 0
        return -1
