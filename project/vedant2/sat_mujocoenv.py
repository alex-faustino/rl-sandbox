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


from gym import core, spaces
from gym.utils import seeding


import mujoco_py
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np
import xml.etree.ElementTree


class Sat_mujocoEnv: #fixed target, now modifying to changing goals for curriculum learning
    
    def __init__(self,t_hor = 1000, maxabs_torque=10,dt = 10, target_state = np.array([1,0,0,0,0,0,0]), w_mag = 2.5e-3 , w_tumble = None, Noise = None,visualize = False):
        
        
        
        self.model = load_model_from_path(os.path.join(os.getcwd(),'Satellite.xml'))
        self.sim = MjSim(self.model)
        if visualize:
            self.viewer = MjViewer(self.sim)
        self.t = 0
        self.dt = dt
        self.step_num = 0
        self.action_dim = 3;
        self.observation_dim = 7;
        self.action_max = maxabs_torque
        self.w_init_mag = w_mag;
       
        self.max_nstep = t_hor;
        self.target = target_state;
        self.q_target = target_state[0:4]
        self.w_target = target_state[4:]
        
        self.q_skew = np.matrix([[self.q_target[0],self.q_target[3],-self.q_target[2],-self.q_target[1]],
                            [-self.q_target[3],self.q_target[0],self.q_target[1],-self.q_target[2]],
                            [self.q_target[2],-self.q_target[1],self.q_target[0],-self.q_target[3]],
                            [self.q_target[1],self.q_target[2],self.q_target[3],self.q_target[0]]])
        
        if (w_tumble == None):
            self.w_tumble = 25.0 * self.w_init_mag;
        else:
            self.w_tumble = w_tumble;
        if (Noise != None):
            self.noise_dim = 6;
        self.x = self.set_init()

        high = np.array([1.0, 1.0, 1.0, 1.0, self.w_tumble, self.w_tumble, self.w_tumble])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        
        high = np.array([self.action_max, self.action_max, self.action_max])
        low = -high
        self.action_space = spaces.Box(low=low, high=high)
        
    def set_init(self,deviation = None):
        if (deviation == None):
            q_init = 2*np.random.rand(4) -1
            q_init = q_init/np.linalg.norm(q_init)
        else:
            q_dev = np.cos(devaition);
            q_dir = 2*np.random.rand(3) -1;
            q_dir = q_dir/np.linalg.norm(q_dir)*(1-q_dev**2);
            q_init = np.concatenate((q_dev,q_dir))
        q_init_e = np.matmul(self.q_skew, q_init);
        w_init = 2*np.random.rand(3)-1
        w_init = (w_init/np.linalg.norm(w_init))*self.w_init_mag
        w_init_e = target_state[0:4] - w_init
        self.sim.data.qpos[3:] = q_init;
        self.sim.data.qvel[3:] = w_init
        self.sim.forward()
        self.x = np.concatenate((q_init_e,w_init_e,w_init))   
        return self.x    
            
    def get_obs(self):
        q = self.sim.data.qpos[3:]
        w = self.sim.data.qvel[3:]
        q_e = np.matmul(self.q_skew, q);
        w_e = target_state[0:4] - w
        self.x = np.concatenate((q_e,w_e,w))
        return self.x
    
    
    '''
    def drift_traj(self):
    '''    
    def detumble(self,obs):
        if( ((np.linalg.norm(obs[4:] ))< (self.w_init_mag/5)) ):# & (np.linalg.norm(obs[0])>np.cos(0.035))
            return 1
        else:
            return 0
    def terminal(self,obs):
        if(self.detumble(obs) & (np.linalg.norm(obs[0])>np.cos(0.035))):
            return 20.0
        else:
            return 0.0
        
    def get_reward(self,obs,action):
        reset = False
        if (np.linalg.norm(obs[4:])>self.w_tumble):
            reset = True
            reward = -2.0e2
            self.reset()

        else:
            if(self.target.ndim ==1):
                reward = 0
                reward += self.terminal(obs)
                reward += 5*((self.step_num/self.max_nstep)**2)*((1*np.linalg.norm(obs[0]))**(2))*self.detumble(obs)# - self.target[0:4]
                '''
                reward += 1*((self.step_num/self.max_nstep)**2)*((1*np.linalg.norm(obs[0]))**(1))# - self.target[0:4]
                reward -= 1*((self.step_num/self.max_nstep)**2)*((10*(np.linalg.norm(obs[4:] ))/self.w_init_mag)**(1))#- self.target[4:]
                #reward += 2*((self.step_num/self.max_nstep)**2)*((1*np.linalg.norm(obs[0]))**(2))# - self.target[0:4]
                reward -= 10*((self.step_num/self.max_nstep)**2)*((10*(np.linalg.norm(obs[4:] ))/self.w_init_mag)**(2))#- self.target[4:]
                '''
                #reward -= 1*((self.step_num/self.max_nstep)**2)*((10*(np.linalg.norm(obs[4:] ))/self.w_init_mag)**(2))/self.max_nstep#- self.target[4:]
                #reward += np.exp(1/((np.linalg.norm(obs[0] )+1e-7)))
                reward += np.exp(1e-6/((np.linalg.norm(obs[4:] ))/self.w_init_mag))
                reward -= 1*np.linalg.norm(action)
            else:
                reward = -1*((np.linalg.norm(obs[0:4]- self.target[self.step_num][0:4])**4))  -1*(1000*(np.linalg.norm(obs[4:] - self.target[self.step_num][4:])**4)) - 1*np.linalg.norm(action)
        #print(reward)
        if(self.step_num > self.max_nstep):
            reset = True
            self.reset()           
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
        #self.a = a
        self.a = np.clip(a,-self.action_max,self.action_max)    
        self.sim.data.ctrl[0] = self.a[0]
        self.sim.data.ctrl[1] = self.a[1]
        self.sim.data.ctrl[2] = self.a[2]
        self.step_num += 1
        self.t += self.dt
        self.sim.step()
        self.x = self.get_obs();
        reward,reset_ = self.get_reward(self.x,self.a)
        if(reset_):
            self.x = self.get_obs();
        if(render):
            self.viewer.render()
        
            
        return self.x, reward ,done , {}
    
    def copy(self):
        c = Sat_mujocoEnv()
        c.s = self.x.copy()
        return c
    
    def reset(self,change_mass_prop = False):
        '''
        if(change_mass_prop):
            sim.model.body_inertia[0][0] = np.clip(np.random.rand() , 0.1,1.0)
            sim.model.body_inertia[0][1] = np.clip(np.random.rand() , 0.1,1.0)
            sim.model.body_inertia[0][2] = np.clip(np.random.rand() , 0.1,1.0)
        '''
        self.x = self.set_init()
        self.step_num = 0
        return self.x
