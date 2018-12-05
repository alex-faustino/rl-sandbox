# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:51:00 2018

@author: vedant2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:46:26 2018

@author: vedant2
"""
import mujoco_py
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np
import xml.etree.ElementTree

class Sat_mujocoEnv:
    
    def __init__(self, maxabs_torque=10,t_horizon = 3600, target_state = np.array([0,0,0,1,0,0,0]), w_mag = 2.5e-3 , w_tumble = None, Noise = None,visualize = False):

        self.model = load_model_from_path(os.path.join(os.getcwd(),'Satellite.xml'))
        self.sim = MjSim(self.model)
        if visualize:
            self.viewer = MjViewer(self.sim)
        self.t = 0
        self.dt = 10
        self.action_dim = 3;
        self.observation_dim = 7;
        self.action_max = maxabs_torque*np.ones([3,1])
        self.w_init_mag = w_mag;
        self.target = target_state
        self.t_hor = t_horizon
        #render = render
        if (w_tumble == None):
            self.w_tumble = self.w_init_mag*2;
        if (Noise != None):
            self.noise_dim = 6;
        self.x = self.set_init()

        
    def set_init(self):
        q_init = np.random.rand(4)
        q_init = q_init/np.linalg.norm(q_init)
        w_init = np.random.rand(3)
        w_init = (w_init/np.linalg.norm(w_init))*self.w_init_mag
        self.sim.data.qpos[3:] = q_init;
        self.sim.data.qvel[3:] = w_init
        return  np.concatenate((q_init,w_init))      
            
    def get_obs(self):
        q = self.sim.data.qpos[3:]
        w = self.sim.data.qvel[3:]

        return np.concatenate((q,w))
    
    def get_reward(self,obs,action):
        reset = False
        if (np.linalg.norm(obs[4:])>self.w_tumble):
            reset = True
            reward = -1e5
            self.reset()
        else:
            reward = np.linalg.norm(obs - self.target) - np.linalg.norm(action)
        return reward,reset
            
        
    def step(self,a,render = False):
        done = False
        '''
        action_noise = len(sim.data.ctrl)
        if (Noise != None):
            action_noise[0] = math.cos(t / 10.) * Dx#math.cos(t / 10.) * 
            action_noise[1] = math.cos(t / 10.) * Dy
            action_noise[2] = math.cos(t / 10.) * Dz
        '''

        self.a = np.clip(a,-self.action_max,self.action_max)    
        self.sim.data.ctrl[0] = a[0]
        self.t += self.dt
        self.sim.step(self.dt)
        self.x = self.get_obs();
        reward,reset_ = self.get_reward(self.x,self.a)
        if self.t >= self.t_hor:
            self.reset()
            return (self.x, 0, done)
        
        if(reset_):
            self.x = self.get_obs();
        if(render):
            self.viewer.render()
        return self.x, reward ,done
    
    def copy(self):
        c = Sat_mujocoEnv()
        c.s = self.x.copy()
        c.t_hor = self.t_hor
        return c
    
    def reset(self):
        self.x = self.set_init()
        return self.x
