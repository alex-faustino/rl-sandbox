import gym
from gym import error, spaces, core
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from gym.envs.classic_control import rendering
def wrap(x, m, M):
    if x > M:
        x = x-M+m
    if x < m:
        x = x+M-m
    return x

def bound(x, m, M):
    return min(max(x,m), M)

def rk4(derivs, y0, t, *args, **kwargs):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)
    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]
        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


class AcroBot(core.Env):
    metadata = {
        'render.modes':['human','rgb_array'],
        'video.frames_per_second':15
    }

    dt = .2
    delta = np.pi/6
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    def __init__(self, M):
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low = low, high = high)
        self.action_space =spaces.Discrete(3)
        self.state = None
        self.M = M
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        s = self.state
        if action == 0:
            torque = 0
        elif action == 1:
            torque = self.M
        elif action == 2:
            torque = -self.M
        else:
            raise Exception(f'invalid action: {action}')
        s_augmented = np.append(s,torque)
        st = rk4(self._dsdt, s_augmented, [0, self.dt])
        st = st[-1]
        st = st[:4]
        st[0] = wrap(st[0], -pi, pi)
        st[1] = wrap(st[1], -pi, pi)
        st[2] = bound(st[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        st[3] = bound(st[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = st
        reward = 1. if self._reward else 0.
        return (self._get(), reward, self._terminal(), {})

    def reset(self):
        self.state = self.np_random.uniform(low = -0.1, high = 0.1, size = (4,))
        return self._get()

    def render(self, mode = 'human', close = False):
        from gym.envs.classic_control import rendering
        s= self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)

        if s is None: return None
        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th) in zip(xys, thetas):
            l,r,t,b = 0, 1, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _get(self):
        s = self.state
        return np.array([s[0], s[1], s[2], s[3]])


    def _terminal(self):
        s = self.state
        #print('EEEEEEE',bool(-cos(s[0])-cos(s[0]+s[1])>1))
        return bool(-cos(s[0])-cos(s[0]+s[1])>1)

    def _reward(self):
        s = self.state
        return bool((np.pi/2-delta)<s[0]<(np.pi/2-delta) and (0-delta)<s[1]<(0-delta))
    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1*lc1**2+m2*(l1**2+lc2**2+2*l1*lc2*cos(theta2))+I1+I2
        d2 = m2*(lc2**2+l1*lc2*cos(theta2))+I2
        phi2 = m2*lc2*g*cos(theta1+theta2-pi/2.)
        phi1 = -m2*l1*lc2*dtheta2**2*sin(theta2)-2*m2*l1*lc2*dtheta2*dtheta1*sin(theta2)+(m1*lc1+m2*l1)*g*cos(theta1-pi/2)+phi2
        ddtheta2 = (a+d2/d1*phi1-m2*l1*lc2*dtheta1**2*sin(theta2)-phi2)/(m2*lc2**2+I2-d2**2/d1)
        ddtheta1 = -(d2*ddtheta2+phi1)/d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)
