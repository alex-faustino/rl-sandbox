"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
"""
In this implementation we focused on the following important changes mentioned in the HW. 
we mentioned these in the code where the design considerations have been implemented
1. the time step is fixed; 
2. the agent can observe joint angles and joint velocities (perfect measurements) at the start of each time step;
3. the control input is constant throughout each time step (i.e., zero-order hold);
4. the reward is +1 when the first joint angle is in the interval [(pi / 2) - delta, (pi / 2) + delta] and 
the second joint angle is in the interval [0 - delta, 0 + delta], and 0 otherwise;
5. the initial joint angles and joint velocities are each sampled from a unit normal distribution about zero.

Basic description of Acrobot: 
Acrobot is a 2-link pendulum with only the second joint actuated
Intitially, both links point downwards. The goal is to swing the
end-effector at a height at least the length of one link above the base.
Both links can swing freely and can pass by each other, i.e., they don't
collide when they have the same angle.
**STATE:**
The state consists of the sin() and cos() of the two rotational joint
angles and the joint angular velocities :
[cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
For the first link, an angle of 0 corresponds to the link pointing downwards.
The angle of the second link is relative to the angle of the first link.
An angle of 0 corresponds to having the same angle between the two links.
"""

class AcroEnvNew(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2
    ## Delta value is used for the calculation of the reward function
    DELTA = pi/3 #

    L1 = 1.  # [m]
    L2 = 1.  # [m]
    M1 = 1.  #: [kg] mass of link 1
    M2 = 1.  #: [kg] mass of link 2
    POS1 = 0.5  #: [m] position of the center of mass of link 1
    POS2 = 0.5  #: [m] position of the center of mass of link 2
    MOI = 1  #: moments of inertia for both links

    MV1 = 10 * np.pi
    MV2 = 20 * np.pi

    Mval = 2.
    AVAIL_TORQUE = [-Mval, 0., Mval]
    torque_noise_max = 0.

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MV1, self.MV2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.info = {}
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        ### 5. the initial joint angles and joint velocities are each sampled from a unit normal distribution about zero.
        self.state = np.random.normal(0, 0.5, (4,)) 
        s = self.state
        self.meas = np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
        return self.meas, self.state
    
        #self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        #return self._get_ob()

    def step(self, act):
        s_old = self.state
        meas_old = np.array([cos(s_old[0]), sin(s_old[0]), cos(s_old[1]), sin(s_old[1]), s_old[2], s_old[3]])
        self.info['cur_out'] = meas_old
        
        ### Add noise to the torque to make it more realistic
        action = self.AVAIL_TORQUE[act]
        if self.torque_noise_max > 0:
            action += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)
        ### State and action are augmented and passed to
        ### equation of motion function _dsdt
        # Now, augment the state with our force action so it can be passed to
        #3. the control input is constant throughout each time step (i.e., zero-order hold);
        y0 = np.append(s_old, action)

        ### RK4 implementation is used for solving the ODE's
        ns = rk4(self._dsdt, y0, [0, self.dt])
        ns = ns[-1]
        ns = ns[:4]

        ### Apply bounds on the solved equations of motion 
        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MV1, self.MV1)
        ns[3] = bound(ns[3], -self.MV2, self.MV2)
        self.state = ns
        
        ### Here s[0]=theta_1, s[1]=theta_2
        ### 2. The agent can observe joint angles and joint velocities (perfect measurements) at the start of each time step;
        s_new = self.state
        self.meas = np.array([cos(s_new[0]), sin(s_new[0]), cos(s_new[1]), sin(s_new[1]), s_new[2], s_new[3]])
        
        ### 4. the reward is +1 when the first joint angle is in the interval [(pi / 2) - delta, (pi / 2) + delta] 
        ### and the second joint angle is in the interval [0 - delta, 0 + delta], and 0 otherwise;
        reward = -self.L1*np.cos(s_new[0]) - self.L2*np.cos(s_new[0] + s_new[1])
        if(reward>0): 
            reward = 3*reward
        #reward = 0.
        #f ( (s_new[0]>= (pi/2.0)-self.DELTA) and (s_new[0]<= (pi/2.0) +self.DELTA) ): 
        #   if( (s_new[1]<=self.DELTA) and (s_new[1]>= -self.DELTA) ): 
        #       reward = 1.
        #reward = -1. if not done else 0.
        
        self.info['next_out'] = self.meas
        return (self.state, reward, 0., self.info)

    ### Equations of motion of the double pendulum/acrobot
    def _dsdt(self, s_augmented, t):
        m1 = self.M1
        m2 = self.M2
        l1 = self.L1
        lc1 = self.POS1
        lc2 = self.POS2
        I1 = self.MOI
        I2 = self.MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
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
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    ### The implementation of rendering has been referenced from  
    ### the openai gym acrobot example
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)

        if s is None: return None

        p1 = [-self.L1 *
              np.cos(s[0]), self.L1 * np.sin(s[0])]

        p2 = [p1[0] - self.L2 * np.cos(s[0] + s[1]),
              p1[1] + self.L2 * np.sin(s[0] + s[1])]

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

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

### The implementation of RK4 has been referenced from  
### the openai gym acrobot example
def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


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
