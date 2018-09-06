# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 01:19:20 2018

@author: Vedant
"""

from gym import core, spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from numpy import sin, cos, pi

import os
import sys
from time import time as Time
from time import sleep

from attitudeDynamics.modules import mod_attitude
import pygame

import pdb;


# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class attitudeDynamics(core.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    mu = 3.986004e14;     # m J /kg
    r_earth = 6378.14e3;  # Earth Radius [m]
    
    # ------------------------------------------------------------------------------
    # CONTROL VARIABLES
    nOrbits = 0.1; #number of orbits
    dt      = 1.0; # time step
    model = 'hiakasat';
    
    #-------------------------------------------------------------------------------
    # INITIALIZATION
    # initial angular speed [deg/s]
    # q is defined with scalar first [q0 q1 q2 q3]
    q_0 = np.array([1.0, 0.0, 0.0, 0.0]);
    omega_0 = np.array([0.0, 0.0, 1.0]);
    
    q_f = np.array([1.0, 0.0, 0.0, 0.0]);
    omega_f = np.array([0.0, 0.0, 0.0]);
    # orbital data
    h_sat = 500e3; # height [m]
    r_sat = r_earth + h_sat; # radius [m]
    v = np.sqrt(mu/r_sat); # speed [m/s]
    P = 2*pi*r_sat/v; #period [s]
    Pminutes = P/60;
    
    tf = nOrbits*P;#sec
    #tf = 1000;
    
    # for gravity gradient stability
    # Iy > Ix > Iz
    I_xx=2.5448;
    I_yy=2.4444;
    I_zz=2.6052;
    
    MAX_VEL = 5*np.pi/180
    #I_xx=1;
    #I_yy=1;
    #I_zz=1;
    
    Inertia =  np.diag([I_xx,I_yy,I_zz]);
    
    
    ##
    # state vector
    s_init = np.concatenate([q_0,omega_0]);
    s_f = np.concatenate([q_f,omega_f]);
    #t  = 0:dt:tf;                                   
    # set time points

    torque_max = 1.0
    delta = 0.0001
    torque_noise_max = 0.

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL, self.MAX_VEL, self.MAX_VEL])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        high = np.array([self.torque_max, self.torque_max, self.torque_max])
        low = -high
        self.action_space = spaces.Box(low=low, high=high)
        self.state = self.s_init
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        q_init = np.array([1.0,0.0,0.0,0.0])
        self.state = np.append(q_init, self.np_random.uniform(low=-self.MAX_VEL, high=self.MAX_VEL, size=(3,)))
        return self.state

    def step(self, a):
        s = self.state
        ns = odeint(mod_attitude.attitude_dynamics, s, [0,self.dt], args=(a,self.Inertia))
        ns =ns[-1]
        #pdb.set_trace()
        # compute control Torque
        # 
        q     = ns[0:4];
        #ns[4] = bound(ns[4], -self.MAX_VEL, self.MAX_VEL)
        #ns[5] = bound(ns[5], -self.MAX_VEL, self.MAX_VEL)
        #ns[6] = bound(ns[6], -self.MAX_VEL, self.MAX_VEL)
        self.state = ns
        terminal = self._terminal()
        reward = 1. if terminal else 0.
        return (s, reward, False, {})

    def _terminal(self):
        s = self.state
        de = np.linalg.norm(np.abs(s-self.s_f))
        return bool(de<self.delta)
    def render(self, mode='human'):
        s = self.state
        q = s[0:4]
        R = mod_attitude.dcm_from_quaternion(q)
        from attitudeDynamics import attitude_viz
        if self.viewer is None:
            self.viewer = attitude_viz.Point3D()
            pygame.init()
        attitude_viz.Simulation.run(np.matrix(R))
        return 0

    def close(self):
        if self.viewer: self.viewer.close()

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
