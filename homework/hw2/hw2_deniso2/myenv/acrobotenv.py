from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

class AcrobotEnv(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    dt = .01

    # parameters are taken from gym acrobot
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links
    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    def __init__(self):
        high = np.array([np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
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
        dth1, dth2, ddth1, ddth2, _ = self._dsdt(self.state, a, self.dt)
        # dsdt is dynamics taken from gym environment
        ns = np.zeros(4)
        ns[0] = self.state[0] + dth1
        ns[1] = self.state[1] + dth2
        ns[2] = self.state[2] + ddth1
        ns[3] = self.state[3] + ddth2

        if ns[0] > np.pi: ns[0] = ns[0] - np.pi
        elif ns[0] < -np.pi: ns[0] = ns[0] + np.pi
        if ns[1] > np.pi: ns[1] = ns[1] - np.pi
        elif ns[1] < -np.pi: ns[1] = ns[1] + np.pi
        ns[2] = max(min(ns[2],10),-10)
        ns[3] = max(min(ns[3],10),-10)

        self.state = ns
        if (ns[0] < np.pi/2 + 0.1 and
            ns[0] > np.pi/2 - 0.1 and
            ns[1] < 0.1 and
            ns[1] > -0.1):
            done = True
            reward = 1
        return (self.state, reward, done, {})

    # this function is taken from acrobot dynamics in gym
    def _dsdt(self, s, a, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2

        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        # my approach for render with matplotlib
        s = self.state

        p1 = [-self.LINK_LENGTH_1 * np.cos(s[0]),
            self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        plt.plot([0,p1[0]], [0,p1[1]], 'b-', lw=5)
        plt.plot([p1[0],p2[0]], [p1[1],p2[1]], 'r-', lw=5)
        plt.axis([-2, 2, -2, 2])
        plt.show()

        return 0
