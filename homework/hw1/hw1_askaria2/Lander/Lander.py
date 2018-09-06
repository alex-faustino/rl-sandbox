"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from time import sleep
import matplotlib.pyplot as plt

__copyright__ = "Copyleft"
__license__ = "None"
__author__ = "Alireza Askarian <askaria2@illinois.edu>"

class LanderEnv(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    #time-step
    dt = .02

    #define system variables
    LANDER_MASS = 1.  #: [kg] mass of lander
    GRAVITY = 9.8  #: gravitational constant

    def __init__(self):
        self.viewer = None
        high = np.array([1000, 200.0])
        low = np.array([0, -200.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(10)
        self.reward = 0
        self.terminal = 0
        self.thrust = 0
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([900, 0])
        self.terminal = 0
        self.reward = 0
        return self._get_ob()

    def step(self, a):

        s = self.state
        self.thrust = a

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, self.thrust)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:2]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt
        
        Fuel = -0.01
        Time = -0.001
        Speed = -0.1
        self.reward = self.reward + Fuel*self.thrust + Time

        self.state = ns
        if self.state[0] <= 100:
            self.terminal = 1
            self.reward = self.reward + Speed*np.absolute(self.state[1])
        return (self._get_ob(), self.reward, self.terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m = self.LANDER_MASS
        g = self.GRAVITY
        a = s_augmented[-1]
        s = s_augmented[:-1]

        dx = s[1]

        #dynamical equations of motion
        ddx = a/m - g
        return (dx, ddx, 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state
        a = np.array(self.thrust)/50

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-5,5,0,10)

        if s is None: return None
        
        self.viewer.draw_line((-5, 1), (5, 1))

        l,r,t,b = -0.1, .1, 1, 0
        jtransform = rendering.Transform(rotation=0, translation=(0,s[0]/100))
        link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
        link.add_attr(jtransform)
        link.set_color(0.8, .8, .8)
        circ = self.viewer.draw_circle(a)
        circ.set_color(1, 0, 0)
        circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

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

env = LanderEnv()

episode_num = 1
time_horizon = 800

reward_vector = np.zeros(time_horizon)
x = np.zeros(time_horizon)
x_dot = np.zeros(time_horizon)
action_vector = np.zeros(time_horizon)
T = np.zeros(time_horizon)

for i_episode in range(episode_num):
    observation = env.reset()
    for t in range(time_horizon):
        env.render()
        action = env.action_space.sample()
        observation, reward, term, x = env.step(action)

        reward_vector[t] = reward
        action_vector[t] = action
        x[t] = observation[0]
        x_dot[t] = observation[1]
        T[t] = t
        print(x[t], x_dot[t],T[t])
        if term:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()

plt.plot(T, reward_vector, 'r')
plt.ylabel('reward')
plt.xlabel('time index')
plt.show()

plt.plot(T, action_vector, 'b')
plt.ylabel('action')
plt.xlabel('time index')
plt.show()

plt.plot(T, x, 'b')
plt.plot(T, x_dot, 'r')
plt.ylabel('x and xdot')
plt.xlabel('time index')
plt.show()
