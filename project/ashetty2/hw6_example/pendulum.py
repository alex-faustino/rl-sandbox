import random, math
import numpy as np

class PendulumEnv:
    def __init__(self, maxabs_tau=20):
        self.observation_dim = 3    # observation is [x, y, omega]
        self.action_dim = 1
        self.params = {
            'dt': 0.1,
            'm': 1.0,
            'g': 9.8,
            'l': 1.0,
            'b': 1.0,
            'maxabs_tau': maxabs_tau,
            'maxabs_omega': 15,
            'maxabs_theta_init': math.pi,
            'maxabs_omega_init': 5,
            'horizon': 200,
        }
        self.reset()

    def s2z(self, s):
        """
        Converts observation (x,y,omega) to state z (theta,omega)
        """
        theta = math.atan2(s[0], -s[1])
        omega = s[2]
        return np.array([theta, omega])

    def z2s(self, z):
        """
        Converts state z (theta,omega) to observation (cos,sin,omega)
        """
        return np.array([math.sin(z[0]), -math.cos(z[0]), z[1]])

    def _dzdt(self, z, tau):
        theta, omega = z[0], z[1]
        p = self.params
        theta_ddot =  (tau - self.params['b'] * omega
                       - self.params['m'] * self.params['g'] * self.params['l'] * np.sin(theta)) \
                       / (self.params['m'] * self.params['l']**2)
        return np.array([omega, theta_ddot])

    def step(self, a):
        done = False
        self.n_steps += 1

        # Check if the horizon has been reached (if so, reset and return with zero reward)
        if self.n_steps >= self.params['horizon']:
            self.reset()
            return (self.s, 0, done)

        # Compute new state
        z = self.s2z(self.s)
        tau = np.clip(a * 10, -self.params['maxabs_tau'], self.params['maxabs_tau'])
        z_new = z + self.params['dt'] * self._dzdt(z, tau)
        self.s = self.z2s(z_new)

        # Check if constraints have been violated (if so, reset and return with large negative reward)
        if abs(self.s[2]) > self.params['maxabs_omega']:
            r = -100
            self.reset()
            return (self.s, r, done)

        # Compute reward
        r = - self.s[0]**2 - (self.s[1] - 1)**2 - 0.01 * self.s[2]**2 - 1 * a**2
        r = r.item()

        return (self.s, r, done)

    def reset(self):
        z = np.random.uniform([-self.params['maxabs_theta_init'], -self.params['maxabs_omega_init']],
                              [self.params['maxabs_theta_init'], self.params['maxabs_omega_init']])
        self.s = self.z2s(z)
        self.n_steps = 0
        return self.s

    def copy(self):
        c = PendulumEnv()
        c.s = self.s.copy()
        c.params = self.params.copy()
        c.n_steps = self.n_steps
        return c

    def render(self):
        pass
