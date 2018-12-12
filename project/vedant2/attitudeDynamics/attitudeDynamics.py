# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 01:19:20 2018

@author: Vedant
"""

from gym import core, spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import ode
from numpy import sin, cos, pi

import os
import sys
from time import time as Time
from time import sleep

from attitudeDynamics.modules import mod_attitude
#import pygame

import pdb;




# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class attitudeDynamics(core.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    #mu = 3.986004e14;     # m J /kg
    #r_earth = 6378.14e3;  # Earth Radius [m]
    
    # ------------------------------------------------------------------------------
    # CONTROL VARIABLES
    #nOrbits = 1; #number of orbits
    dt      = 1.0; # time step
    #model = 'hiakasat';
    
    #-------------------------------------------------------------------------------
    # INITIALIZATION
    # initial angular speed [deg/s]
    # q is defined with scalar first [q0 q1 q2 q3]
    q_0 = np.array([1.0, 0.0, 0.0, 0.0]);
    omega_0 = np.array([5*np.pi*np.random.rand()/180, 5*np.pi*np.random.rand()/180, 5*np.pi*np.random.rand()/180]);
    
    q_f = np.array([1.0, 0.0, 0.0, 0.0]);
    omega_f = np.array([0.0, 0.0, 0.0]);
    # orbital data
    #h_sat = 500e3; # height [m]
    #r_sat = r_earth + h_sat; # radius [m]
    #v = np.sqrt(mu/r_sat); # speed [m/s]
    #P = 2*pi*r_sat/v; #period [s]
    #Pminutes = P/60;
    
    #tf = nOrbits*P;#sec
    #tf = 1000;
    
    # for gravity gradient stability
    # Iy > Ix > Iz
    I_xx=2.5448+0.5*np.random.rand();
    I_yy=2.4444+0.5*np.random.rand();
    I_zz=2.6052+0.5*np.random.rand();
    
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

    torque_max = 1#5e-1
    delta = 0.001
    torque_noise_max = 0.

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL, self.MAX_VEL, self.MAX_VEL])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.observation_dim = 7
        self.action_dim = 3
        high = np.array([self.torque_max, self.torque_max, self.torque_max])
        low = -high
        self.action_space = spaces.Box(low=low, high=high)
        self.state = np.append(self.q_0,np.random.uniform(low=-self.MAX_VEL, high=self.MAX_VEL, size=(3,)))
        self.ns = ode(mod_attitude.attitude_dynamics).set_integrator('dop853', nsteps = 1000000)#, method='bdf'
        self.ns.set_initial_value(self.state,0)
        #self.reset()
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        
        self.I_xx=2.5448+0.5*np.random.rand();
        self.I_yy=2.4444+0.5*np.random.rand();
        self.I_zz=2.6052+0.5*np.random.rand();
        self.state = np.append(self.q_0, np.random.uniform(low=-self.MAX_VEL, high=self.MAX_VEL, size=(3,)))
        del(self.ns)
        #print("ns deleted should reset")
        self.ns = ode(mod_attitude.attitude_dynamics).set_integrator('dop853')
        self.ns.set_initial_value(self.state,0)
        #print(self.ns.t)
        return self.state

    def step(self, a):
        s = self.state
        #ns = ode(mod_attitude.attitude_dynamics, s, [0,self.dt], args=(a,self.Inertia))
        #print(np.clip(a,-1*self.torque_max,self.torque_max))
        self.ns.set_f_params(np.clip(a,-1*self.torque_max,self.torque_max),self.Inertia)
        ns_d  = self.ns.integrate(self.ns.t+self.dt)
        #print(ns_d)
        #ns_d =ns_d[-1]
        #pdb.set_trace()
        # compute control Torque
        # 
        q     = ns_d[0:4];
        #ns[4] = bound(ns[4], -self.MAX_VEL, self.MAX_VEL)
        #ns[5] = bound(ns[5], -self.MAX_VEL, self.MAX_VEL)
        #ns[6] = bound(ns[6], -self.MAX_VEL, self.MAX_VEL)
        self.state = ns_d
        terminal = self._terminal()
        reward = (-1*np.linalg.norm(ns_d[4:7]))+1*(q[0]) -np.abs(q[1])-np.abs(q[2])-np.abs(q[3]) -5*np.linalg.norm(a)#+1000*terminal
        return (s, reward, False, 0)

    def _terminal(self):
        s = self.state
        de = np.linalg.norm(np.abs(s-self.s_f))
        return bool(de<self.delta)
    def render(self, mode='human'):
        #s = self.state
        #q = s[0:4]
        #R = mod_attitude.dcm_from_quaternion(q)
        #from attitudeDynamics import attitude_viz
        #if self.viewer is None:
            #self.viewer = attitude_viz.Point3D()
            #pygame.init()
        #attitude_viz.Simulation.run(np.matrix(R))
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
