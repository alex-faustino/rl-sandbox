from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

class AcrobotEnv(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    dt = .1

    def __init__(self):
        high = np.array([np.pi, np.pi, 5*np.pi, 10*np.pi])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.normal(0, 1, 4)
        return self.state

    def step(self, a):

        done = False
        reward = 0

        # compute derivatives with given state and action
        if a == 1: u = 1
        elif a == 2: u = -1
        else: u = 0

        dq1, dq2, ddq1, ddq2 = self._dsdt(self.state, u, self.dt)

        # apply derivatives to system state
        ns = np.zeros(4)
        ns[0] = self.state[0] + dq1*self.dt
        ns[1] = self.state[1] + dq2*self.dt
        ns[2] = self.state[2] + ddq1*self.dt
        ns[3] = self.state[3] + ddq2*self.dt

        while ns[0] > np.pi: ns[0] = ns[0] - np.pi
        while ns[0] < -np.pi: ns[0] = ns[0] + np.pi
        while ns[1] > np.pi: ns[1] = ns[1] - np.pi
        while ns[1] < -np.pi: ns[1] = ns[1] + np.pi

        ns[2] = max(min(ns[2],5),-5)
        ns[3] = max(min(ns[3],10),-10)

        self.state = ns[:]
        if (abs(ns[0]) > np.pi - 0.1 and
            abs(ns[1]) < 0.1):
            #done = True
            reward = 1
        return (self.state, reward, done, {})

    def _dsdt(self, s, u, dt):
        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        lc1 = 0.5
        lc2 = 0.5
        I1 = 1.0
        I2 = 1.0
        g = 9.8

        q1 = s[0]
        q2 = s[1]
        dq1 = s[2]
        dq2 = s[3]

        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*np.cos(q2),
            I2 + m2*l1*lc2*np.cos(q2)],
            [I2 + m2*l1*lc2*np.cos(q2), I2]])

        C = np.array([[-2*m2*l1*lc2*np.sin(q2)*dq2,
            -m2*l1*lc2*np.sin(q2)*dq2],
            [m2*l1*lc2*np.sin(q2)*dq1, 0]])

        tao = np.array([-m1*g*lc1*np.sin(q1) - m2*g*(l1*np.sin(q1) + lc2*np.sin(q1+q2)),
            -m2*g*lc2*np.sin(q1+q2)])

        ddq1, ddq2 = np.matmul(np.linalg.inv(M),(tao + [0, u] - np.matmul(C, [dq1,dq2])))

        dq1 += ddq1*dt
        dq2 += ddq2*dt

        return (dq1, dq2, ddq1, ddq2)

    def render(self, mode='human'):
        # my approach for render with matplotlib
        s = self.state

        p1 = [1 * np.sin(s[0]), -1 * np.cos(s[0])]

        p2 = [p1[0] + 1 * np.sin(s[0] + s[1]),
              p1[1] - 1 * np.cos(s[0] + s[1])]

        plt.figure()
        plt.axis([-2, 2, -2, 2])
        self.line1, = plt.plot([0, p1[0]], [0, p1[1]], 'b-o', lw=2)
        self.line2, = plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-o', lw=2)
        plt.show()

        return 0
